/*
 * Host media back end for the AvPlayer and Videodec2 HLE modules.
 *
 * A deliberately small C surface over FFmpeg so the Rust side does not
 * depend on FFmpeg struct layouts: container demux + decode of one video
 * and one audio stream (psm_*), and a raw access-unit video decoder
 * (psv_*). Video is delivered as NV12, audio as interleaved s16.
 */
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/buffer.h>
#include <libavutil/channel_layout.h>
#include <libavutil/mathematics.h>
#include <libavutil/mem.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------ */
/* Shared helpers                                                           */
/* ------------------------------------------------------------------------ */

/* Copy (or convert) `frame` into an NV12 surface: luma at dst with `pitch`
 * bytes per row and `rows` rows, chroma right after it. Rows and columns
 * beyond the picture are filled with black (when `pad` is set) so samplers
 * reading the padding see no colour fringe. */
static int write_nv12(struct SwsContext **sws, const AVFrame *frame, uint8_t *dst, uint32_t pitch,
                      uint32_t rows, int full_range, int pad) {
    const int w = frame->width, h = frame->height;
    if (w <= 0 || h <= 0 || (uint32_t)w > pitch || (uint32_t)h > rows) return -1;
    uint8_t *chroma = dst + (size_t)pitch * rows;
    const uint32_t chroma_rows = (rows + 1) / 2;
    const uint8_t black = full_range ? 0 : 16;
    uint32_t cw = (uint32_t)((w + 1) & ~1);
    if (cw > pitch) cw = pitch;

    if (frame->format == AV_PIX_FMT_NV12) {
        for (int y = 0; y < h; y++)
            memcpy(dst + (size_t)y * pitch, frame->data[0] + (ptrdiff_t)y * frame->linesize[0], (size_t)w);
        for (int y = 0; y < (h + 1) / 2; y++)
            memcpy(chroma + (size_t)y * pitch, frame->data[1] + (ptrdiff_t)y * frame->linesize[1], cw);
    } else {
        *sws = sws_getCachedContext(*sws, w, h, (enum AVPixelFormat)frame->format, w, h, AV_PIX_FMT_NV12,
                                    SWS_FAST_BILINEAR, NULL, NULL, NULL);
        if (!*sws) return -1;
        uint8_t *planes[4] = {dst, chroma, NULL, NULL};
        int strides[4] = {(int)pitch, (int)pitch, 0, 0};
        if (sws_scale(*sws, (const uint8_t *const *)frame->data, frame->linesize, 0, h, planes, strides) < 0)
            return -1;
    }

    if (!pad) return 0;
    if ((uint32_t)w < pitch) {
        for (int y = 0; y < h; y++) memset(dst + (size_t)y * pitch + w, black, pitch - (uint32_t)w);
        for (int y = 0; y < (h + 1) / 2; y++) memset(chroma + (size_t)y * pitch + cw, 128, pitch - cw);
    }
    for (uint32_t y = (uint32_t)h; y < rows; y++) memset(dst + (size_t)y * pitch, black, pitch);
    for (uint32_t y = (uint32_t)(h + 1) / 2; y < chroma_rows; y++) memset(chroma + (size_t)y * pitch, 128, pitch);
    return 0;
}

/* ------------------------------------------------------------------------ */
/* Container playback                                                       */
/* ------------------------------------------------------------------------ */

/* All container I/O goes through caller-provided callbacks (opaque, read,
 * seek) with AVIO semantics, so the caller decides where bytes come from. */
typedef int (*psm_read_fn)(void *opaque, uint8_t *buf, int len);
typedef int64_t (*psm_seek_fn)(void *opaque, int64_t offset, int whence);

AVIOContext *psm_io_new(void *opaque, psm_read_fn rd, psm_seek_fn sk, int bufsize) {
    uint8_t *buf = av_malloc((size_t)bufsize);
    if (!buf) return NULL;
    AVIOContext *io = avio_alloc_context(buf, bufsize, 0, opaque, rd, NULL, sk);
    if (!io) av_free(buf);
    return io;
}

void psm_io_free(AVIOContext *io) {
    if (!io) return;
    av_freep(&io->buffer);
    avio_context_free(&io);
}

/* Buffered access through an AVIO context (avio_read / avio_seek). */
int psm_io_read(AVIOContext *io, uint8_t *buf, int len) { return avio_read(io, buf, len); }
int64_t psm_io_seek(AVIOContext *io, int64_t offset, int whence) { return avio_seek(io, offset, whence); }

typedef struct {
    int32_t media_type; /* 0 video, 1 audio, 2 subtitle, 3 other */
    int32_t decodable;
    int32_t webm_ok; /* codec allowed in a WebM source */
    int32_t width, height;
    int32_t channels, sample_rate, frame_size;
    double fps;
    float sar; /* sample aspect ratio, 0 when unknown */
    int32_t full_range;
    uint32_t color_primaries, color_trc;
    uint64_t duration_ms;
    char lang[4];
} PsmStreamInfo;

typedef struct {
    int32_t width, height;
    uint64_t ts_ms;
    float sar;
    int32_t full_range;
    int32_t crop_left, crop_right, crop_top, crop_bottom;
    int32_t nb_samples, channels, sample_rate;
} PsmFrame;

typedef struct PsmMedia {
    AVFormatContext *fmt;
    AVIOContext *io;
} PsmMedia;

void psm_close(PsmMedia *m) {
    if (!m) return;
    avformat_close_input(&m->fmt);
    psm_io_free(m->io);
    free(m);
}

/* `name` only helps format probing; it may be NULL. */
PsmMedia *psm_open(const char *name, void *opaque, psm_read_fn rd, psm_seek_fn sk, int bufsize, char *err,
                   int errlen) {
    PsmMedia *m = calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->fmt = avformat_alloc_context();
    m->io = psm_io_new(opaque, rd, sk, bufsize);
    if (!m->fmt || !m->io) {
        snprintf(err, (size_t)errlen, "out of memory");
        psm_close(m);
        return NULL;
    }
    m->fmt->pb = m->io;
    m->fmt->flags |= AVFMT_FLAG_CUSTOM_IO;
    int r = avformat_open_input(&m->fmt, name, NULL, NULL);
    if (r < 0) {
        av_strerror(r, err, (size_t)errlen);
        m->fmt = NULL; /* freed by avformat_open_input on failure */
        psm_close(m);
        return NULL;
    }
    r = avformat_find_stream_info(m->fmt, NULL);
    if (r < 0) {
        av_strerror(r, err, (size_t)errlen);
        psm_close(m);
        return NULL;
    }
    return m;
}

int psm_stream_count(const PsmMedia *m) { return (int)m->fmt->nb_streams; }

/* Container duration in ms (0 when unknown). */
uint64_t psm_duration_ms(const PsmMedia *m) { return m->fmt->duration > 0 ? (uint64_t)(m->fmt->duration / 1000) : 0; }

static uint64_t ms_or_zero(int64_t v, AVRational tb) {
    if (v == AV_NOPTS_VALUE) return 0;
    int64_t ms = av_rescale_q(v, tb, (AVRational){1, 1000});
    return ms > 0 ? (uint64_t)ms : 0;
}

int psm_stream_info(const PsmMedia *m, int index, PsmStreamInfo *out) {
    memset(out, 0, sizeof(*out));
    if (index < 0 || (unsigned)index >= m->fmt->nb_streams) return -1;
    const AVStream *s = m->fmt->streams[index];
    const AVCodecParameters *p = s->codecpar;
    enum AVCodecID id = p->codec_id;
    out->webm_ok = 1;
    switch (p->codec_type) {
    case AVMEDIA_TYPE_VIDEO:
        out->media_type = 0;
        out->webm_ok = id == AV_CODEC_ID_VP8 || id == AV_CODEC_ID_VP9 || id == AV_CODEC_ID_AV1;
        break;
    case AVMEDIA_TYPE_AUDIO:
        out->media_type = 1;
        out->webm_ok = id == AV_CODEC_ID_OPUS || id == AV_CODEC_ID_VORBIS;
        break;
    case AVMEDIA_TYPE_SUBTITLE: out->media_type = 2; break;
    default: out->media_type = 3; break;
    }
    out->decodable = avcodec_find_decoder(id) != NULL;
    out->width = p->width;
    out->height = p->height;
    out->channels = p->ch_layout.nb_channels;
    out->sample_rate = p->sample_rate;
    out->frame_size = p->frame_size;
    AVRational fr = s->avg_frame_rate.num != 0 ? s->avg_frame_rate : s->r_frame_rate;
    out->fps = fr.den != 0 ? av_q2d(fr) : 0.0;
    AVRational sar = s->sample_aspect_ratio;
    out->sar = sar.num > 0 && sar.den > 0 ? (float)av_q2d(sar) : 0.0f;
    out->full_range = p->color_range == AVCOL_RANGE_JPEG;
    out->color_primaries = (uint32_t)p->color_primaries;
    out->color_trc = (uint32_t)p->color_trc;
    out->duration_ms = s->duration != AV_NOPTS_VALUE ? ms_or_zero(s->duration, s->time_base) : psm_duration_ms(m);
    const AVDictionaryEntry *e = av_dict_get(s->metadata, "language", NULL, 0);
    if (e && e->value)
        strncpy(out->lang, e->value, 3);
    else
        memcpy(out->lang, "und", 3);
    return 0;
}

/* Drop packets of streams nobody decodes (-1 = none). */
void psm_set_discard(PsmMedia *m, int video, int audio) {
    for (unsigned i = 0; i < m->fmt->nb_streams; i++)
        m->fmt->streams[i]->discard = ((int)i == video || (int)i == audio) ? AVDISCARD_DEFAULT : AVDISCARD_ALL;
}

/* Seek to `ms` on `stream` (-1 = default stream). */
int psm_seek(PsmMedia *m, int stream, uint64_t ms) {
    int64_t ts = stream >= 0 ? av_rescale_q((int64_t)ms, (AVRational){1, 1000}, m->fmt->streams[stream]->time_base)
                             : (int64_t)ms * 1000;
    return avformat_seek_file(m->fmt, stream, INT64_MIN, ts, INT64_MAX, 0);
}

/* Read the next packet. Returns 0 with *out set, or an AVERROR code. */
int psm_read(PsmMedia *m, AVPacket **out, int *stream, int *size) {
    AVPacket *p = av_packet_alloc();
    if (!p) return AVERROR(ENOMEM);
    int r = av_read_frame(m->fmt, p);
    if (r < 0) {
        av_packet_free(&p);
        return r;
    }
    *out = p;
    *stream = p->stream_index;
    *size = p->size;
    return 0;
}

void psm_packet_free(AVPacket *p) { av_packet_free(&p); }

typedef struct PsmDecoder {
    AVCodecContext *ctx;
    AVFrame *frame;
    AVRational tb;
    struct SwsContext *sws;
    SwrContext *swr;
    AVChannelLayout swr_layout;
    int swr_fmt, swr_rate;
    uint8_t *pcm;
    unsigned pcm_cap;
} PsmDecoder;

void psm_decoder_close(PsmDecoder *d) {
    if (!d) return;
    avcodec_free_context(&d->ctx);
    av_frame_free(&d->frame);
    sws_freeContext(d->sws);
    swr_free(&d->swr);
    av_channel_layout_uninit(&d->swr_layout);
    av_free(d->pcm);
    free(d);
}

PsmDecoder *psm_decoder_open(const PsmMedia *m, int index) {
    if (index < 0 || (unsigned)index >= m->fmt->nb_streams) return NULL;
    const AVStream *s = m->fmt->streams[index];
    const AVCodec *codec = avcodec_find_decoder(s->codecpar->codec_id);
    if (!codec) return NULL;
    PsmDecoder *d = calloc(1, sizeof(*d));
    if (!d) return NULL;
    d->swr_fmt = -1;
    d->tb = s->time_base;
    d->ctx = avcodec_alloc_context3(codec);
    d->frame = av_frame_alloc();
    if (!d->ctx || !d->frame || avcodec_parameters_to_context(d->ctx, s->codecpar) < 0) {
        psm_decoder_close(d);
        return NULL;
    }
    d->ctx->pkt_timebase = s->time_base;
    if (s->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
        d->ctx->thread_count = 0;
        d->ctx->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
    }
    if (avcodec_open2(d->ctx, codec, NULL) < 0) {
        psm_decoder_close(d);
        return NULL;
    }
    return d;
}

void psm_decoder_params(const PsmDecoder *d, int *channels, int *frame_size) {
    *channels = d->ctx->ch_layout.nb_channels;
    *frame_size = d->ctx->frame_size;
}

void psm_decoder_flush(PsmDecoder *d) { avcodec_flush_buffers(d->ctx); }

/* avcodec_send_packet; `p` NULL enters draining. */
int psm_decoder_send(PsmDecoder *d, const AVPacket *p) { return avcodec_send_packet(d->ctx, p); }

/* avcodec_receive_frame; on 0 the frame is described in `out` and stays
 * current until the next call. */
int psm_decoder_receive(PsmDecoder *d, PsmFrame *out) {
    av_frame_unref(d->frame);
    int r = avcodec_receive_frame(d->ctx, d->frame);
    if (r < 0) return r;
    const AVFrame *f = d->frame;
    memset(out, 0, sizeof(*out));
    out->ts_ms = ms_or_zero(f->best_effort_timestamp != AV_NOPTS_VALUE ? f->best_effort_timestamp : f->pts, d->tb);
    out->width = f->width;
    out->height = f->height;
    AVRational sar = f->sample_aspect_ratio;
    out->sar = sar.num > 0 && sar.den > 0 ? (float)av_q2d(sar) : 0.0f;
    out->full_range = f->color_range == AVCOL_RANGE_JPEG;
    out->crop_left = (int32_t)f->crop_left;
    out->crop_right = (int32_t)f->crop_right;
    out->crop_top = (int32_t)f->crop_top;
    out->crop_bottom = (int32_t)f->crop_bottom;
    out->nb_samples = f->nb_samples;
    out->channels = f->ch_layout.nb_channels;
    out->sample_rate = f->sample_rate;
    return 0;
}

/* Write the current picture as NV12 (`pitch` x `rows` luma, chroma after);
 * padding is black in the stream's range. */
int psm_decoder_nv12(PsmDecoder *d, uint8_t *dst, uint32_t pitch, uint32_t rows, int stream_full_range) {
    if (!d->frame->data[0]) return -1;
    return write_nv12(&d->sws, d->frame, dst, pitch, rows, stream_full_range, 1);
}

/* The current audio frame as interleaved s16. Returns samples per channel
 * and points *out at them (valid until the next receive). */
int psm_decoder_s16(PsmDecoder *d, const int16_t **out) {
    const AVFrame *f = d->frame;
    if (!f->data[0] || f->nb_samples <= 0) return -1;
    if (f->format == AV_SAMPLE_FMT_S16) {
        *out = (const int16_t *)f->data[0];
        return f->nb_samples;
    }
    if (!d->swr || d->swr_fmt != f->format || d->swr_rate != f->sample_rate ||
        av_channel_layout_compare(&d->swr_layout, &f->ch_layout) != 0) {
        swr_free(&d->swr);
        av_channel_layout_uninit(&d->swr_layout);
        if (swr_alloc_set_opts2(&d->swr, &f->ch_layout, AV_SAMPLE_FMT_S16, f->sample_rate, &f->ch_layout,
                                (enum AVSampleFormat)f->format, f->sample_rate, 0, NULL) < 0 ||
            swr_init(d->swr) < 0) {
            swr_free(&d->swr);
            return -1;
        }
        av_channel_layout_copy(&d->swr_layout, &f->ch_layout);
        d->swr_fmt = f->format;
        d->swr_rate = f->sample_rate;
    }
    int channels = f->ch_layout.nb_channels > 0 ? f->ch_layout.nb_channels : 1;
    av_fast_malloc(&d->pcm, &d->pcm_cap, (size_t)f->nb_samples * (size_t)channels * 2);
    if (!d->pcm) return -1;
    uint8_t *o = d->pcm;
    int n = swr_convert(d->swr, &o, f->nb_samples, (const uint8_t **)f->extended_data, f->nb_samples);
    if (n < 0) return -1;
    *out = (const int16_t *)d->pcm;
    return n;
}

/* ------------------------------------------------------------------------ */
/* Raw access-unit decoder                                                  */
/* ------------------------------------------------------------------------ */

typedef struct {
    uint64_t pts, dts, attached;
} PsvMeta;

typedef struct {
    int32_t width, height, pitch;
    int32_t crop_left, crop_right, crop_top, crop_bottom;
    int32_t profile, level;
    int32_t sar_num, sar_den;
    int32_t color_range, color_primaries, color_trc, color_space;
    int32_t key_frame, corrupt;
    uint64_t pts, dts, attached;
} PsvFrame;

typedef struct PsvDecoder {
    AVCodecContext *ctx;
    AVPacket *pkt;
    AVFrame *frame;
    struct SwsContext *sws;
    /* Set once Flush has sent the end of stream; a reset (or the next access
     * unit) restarts the decoder. */
    int draining;
} PsvDecoder;

void psv_close(PsvDecoder *v) {
    if (!v) return;
    avcodec_free_context(&v->ctx);
    av_packet_free(&v->pkt);
    av_frame_free(&v->frame);
    sws_freeContext(v->sws);
    free(v);
}

/* codec: 0 = H.264, 1 = HEVC, 2 = VP9 */
PsvDecoder *psv_open(int codec) {
    enum AVCodecID id = codec == 0 ? AV_CODEC_ID_H264 : codec == 1 ? AV_CODEC_ID_HEVC : AV_CODEC_ID_VP9;
    const AVCodec *c = avcodec_find_decoder(id);
    if (!c) return NULL;
    PsvDecoder *v = calloc(1, sizeof(*v));
    if (!v) return NULL;
    v->ctx = avcodec_alloc_context3(c);
    v->pkt = av_packet_alloc();
    v->frame = av_frame_alloc();
    if (!v->ctx || !v->pkt || !v->frame) {
        psv_close(v);
        return NULL;
    }
    v->ctx->flags |= AV_CODEC_FLAG_COPY_OPAQUE;
    if (avcodec_open2(v->ctx, c, NULL) < 0) {
        psv_close(v);
        return NULL;
    }
    return v;
}

static void psv_fill(PsvDecoder *v, PsvFrame *out) {
    const AVFrame *f = v->frame;
    memset(out, 0, sizeof(*out));
    out->width = f->width;
    out->height = f->height;
    out->pitch = (f->width + 255) & ~255;
    out->crop_left = (int32_t)f->crop_left;
    out->crop_right = (int32_t)f->crop_right;
    out->crop_top = (int32_t)f->crop_top;
    out->crop_bottom = (int32_t)f->crop_bottom;
    out->profile = v->ctx->profile > 0 ? v->ctx->profile : 0;
    out->level = v->ctx->level > 0 ? v->ctx->level : 0;
    out->sar_num = f->sample_aspect_ratio.num;
    out->sar_den = f->sample_aspect_ratio.den;
    out->color_range = (int32_t)f->color_range;
    out->color_primaries = (int32_t)f->color_primaries;
    out->color_trc = (int32_t)f->color_trc;
    out->color_space = (int32_t)f->colorspace;
    out->key_frame = (f->flags & AV_FRAME_FLAG_KEY) != 0;
    out->corrupt = (f->flags & AV_FRAME_FLAG_CORRUPT) != 0;
    if (f->opaque_ref && f->opaque_ref->size >= sizeof(PsvMeta)) {
        PsvMeta meta;
        memcpy(&meta, f->opaque_ref->data, sizeof(meta));
        out->pts = meta.pts;
        out->dts = meta.dts;
        out->attached = meta.attached;
    } else {
        out->pts = f->pts == AV_NOPTS_VALUE ? UINT64_MAX : (uint64_t)f->pts;
        out->dts = f->pkt_dts == AV_NOPTS_VALUE ? UINT64_MAX : (uint64_t)f->pkt_dts;
    }
}

/* Feed one access unit. Returns 1 with a picture available, 0 without,
 * -1 on an API failure and -2 when the decoder rejected the AU. */
int psv_decode(PsvDecoder *v, const uint8_t *data, int64_t size, uint64_t pts, uint64_t dts, uint64_t attached,
               PsvFrame *out) {
    av_frame_unref(v->frame);
    if (size <= 0 || size > INT_MAX) return -2;
    if (v->draining) {
        avcodec_flush_buffers(v->ctx);
        v->draining = 0;
    }
    if (av_new_packet(v->pkt, (int)size) < 0) return -1;
    memcpy(v->pkt->data, data, (size_t)size);
    v->pkt->pts = pts > (uint64_t)INT64_MAX ? AV_NOPTS_VALUE : (int64_t)pts;
    v->pkt->dts = dts > (uint64_t)INT64_MAX ? AV_NOPTS_VALUE : (int64_t)dts;
    v->pkt->opaque_ref = av_buffer_alloc(sizeof(PsvMeta));
    if (!v->pkt->opaque_ref) {
        av_packet_unref(v->pkt);
        return -1;
    }
    PsvMeta meta = {pts, dts, attached};
    memcpy(v->pkt->opaque_ref->data, &meta, sizeof(meta));

    int have = 0;
    int r = avcodec_send_packet(v->ctx, v->pkt);
    if (r == AVERROR(EAGAIN)) {
        if (avcodec_receive_frame(v->ctx, v->frame) < 0) {
            av_packet_unref(v->pkt);
            return -2;
        }
        have = 1;
        r = avcodec_send_packet(v->ctx, v->pkt);
    }
    av_packet_unref(v->pkt);
    if (r < 0) return -2;
    if (!have) {
        r = avcodec_receive_frame(v->ctx, v->frame);
        if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) return 0;
        if (r < 0) return -2;
    }
    psv_fill(v, out);
    return 1;
}

/* Drain the decoder: the first call ends the stream, every call returns
 * one of the remaining pictures (0 once none is left). */
int psv_flush(PsvDecoder *v, PsvFrame *out) {
    av_frame_unref(v->frame);
    if (!v->draining) {
        int s = avcodec_send_packet(v->ctx, NULL);
        if (s < 0 && s != AVERROR_EOF) return -1;
        v->draining = 1;
    }
    int r = avcodec_receive_frame(v->ctx, v->frame);
    if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) return 0;
    if (r < 0) return -1;
    psv_fill(v, out);
    return 1;
}

void psv_reset(PsvDecoder *v) {
    av_frame_unref(v->frame);
    avcodec_flush_buffers(v->ctx);
    v->draining = 0;
}

/* Write the current picture as NV12: pitch = width aligned to 256, chroma
 * right after `height` luma rows. Returns 0, or -3 if `size` is too small. */
int psv_copy_nv12(PsvDecoder *v, uint8_t *dst, uint64_t size) {
    const AVFrame *f = v->frame;
    if (!f->data[0] || f->width <= 0 || f->height <= 0) return -1;
    uint32_t pitch = ((uint32_t)f->width + 255u) & ~255u;
    uint32_t h = (uint32_t)f->height;
    uint64_t need = (uint64_t)pitch * h + (uint64_t)pitch * ((h + 1) / 2);
    if (need > size) return -3;
    memset(dst, 0, (size_t)need);
    return write_nv12(&v->sws, f, dst, pitch, h, 0, 0);
}

/* ------------------------------------------------------------------------ */
/* Raw audio packet decoder (AJM / Audiodec)                                */
/* ------------------------------------------------------------------------ */

typedef struct {
    int32_t channels, sample_rate, nb_samples;
    int64_t bit_rate;
    /* AVCodecContext.profile of the stream (AAC: 1 LC, 4 HE, 28 HE v2). */
    int32_t profile;
} PsaFrame;

typedef struct PsaDecoder {
    AVCodecContext *ctx;
    AVPacket *pkt;
    AVFrame *frame;
    SwrContext *swr;
    AVChannelLayout swr_layout;
    int swr_in_fmt, swr_out_fmt, swr_rate, swr_out_rate;
    /* Output sample rate; 0 keeps the decoder's rate. */
    int out_rate;
    uint8_t *buf;
    size_t buf_size;
} PsaDecoder;

void psa_close(PsaDecoder *a) {
    if (!a) return;
    avcodec_free_context(&a->ctx);
    av_packet_free(&a->pkt);
    av_frame_free(&a->frame);
    swr_free(&a->swr);
    av_channel_layout_uninit(&a->swr_layout);
    free(a->buf);
    free(a);
}

/* codec: 0 = AAC, 1 = Opus. channels / sample_rate 0 = unset. `extradata`
 * (may be NULL) is the codec's stream header, e.g. an OpusHead. */
PsaDecoder *psa_open_ex(int codec, int channels, int sample_rate, const uint8_t *extradata, int extradata_size) {
    enum AVCodecID id = codec == 1 ? AV_CODEC_ID_OPUS : AV_CODEC_ID_AAC;
    const AVCodec *c = avcodec_find_decoder(id);
    if (!c) return NULL;
    PsaDecoder *a = calloc(1, sizeof(*a));
    if (!a) return NULL;
    a->ctx = avcodec_alloc_context3(c);
    a->pkt = av_packet_alloc();
    a->frame = av_frame_alloc();
    if (!a->ctx || !a->pkt || !a->frame) {
        psa_close(a);
        return NULL;
    }
    if (channels > 0) av_channel_layout_default(&a->ctx->ch_layout, channels);
    if (sample_rate > 0) a->ctx->sample_rate = sample_rate;
    if (extradata && extradata_size > 0) {
        a->ctx->extradata = av_mallocz((size_t)extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
        if (!a->ctx->extradata) {
            psa_close(a);
            return NULL;
        }
        memcpy(a->ctx->extradata, extradata, (size_t)extradata_size);
        a->ctx->extradata_size = extradata_size;
    }
    if (avcodec_open2(a->ctx, c, NULL) < 0) {
        psa_close(a);
        return NULL;
    }
    a->swr_in_fmt = -1;
    return a;
}

PsaDecoder *psa_open(int codec, int channels, int sample_rate) { return psa_open_ex(codec, channels, sample_rate, NULL, 0); }

/* Resample every following frame to `rate` (0 = the decoder's rate). */
void psa_set_output_rate(PsaDecoder *a, int rate) { a->out_rate = rate > 0 ? rate : 0; }

void psa_flush(PsaDecoder *a) {
    av_frame_unref(a->frame);
    avcodec_flush_buffers(a->ctx);
}

/* Feed one packet, optionally with new extradata (AudioSpecificConfig).
 * Returns 0, -1 on an API failure, -2 for invalid data and -3 for other
 * decoder errors. */
int psa_send(PsaDecoder *a, const uint8_t *data, int size, const uint8_t *extradata, int extradata_size) {
    if (size <= 0) return -3;
    if (av_new_packet(a->pkt, size) < 0) return -1;
    memcpy(a->pkt->data, data, (size_t)size);
    if (extradata && extradata_size > 0) {
        uint8_t *side = av_packet_new_side_data(a->pkt, AV_PKT_DATA_NEW_EXTRADATA, (size_t)extradata_size);
        if (!side) {
            av_packet_unref(a->pkt);
            return -1;
        }
        memcpy(side, extradata, (size_t)extradata_size);
    }
    int r = avcodec_send_packet(a->ctx, a->pkt);
    av_packet_unref(a->pkt);
    if (r == AVERROR_INVALIDDATA) return -2;
    return r < 0 ? -3 : 0;
}

/* Collect one decoded frame converted to interleaved fmt (0 = s16,
 * 1 = s32, 2 = float); the samples stay readable through psa_data until
 * the next call. The converter and the buffer are reused between frames.
 * Returns 1 with a frame, 0 when none is ready, -1 on an API failure and
 * -2 for invalid data. */
int psa_receive(PsaDecoder *a, int fmt, PsaFrame *out) {
    av_frame_unref(a->frame);
    int r = avcodec_receive_frame(a->ctx, a->frame);
    if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) return 0;
    if (r < 0) return -2;
    const AVFrame *f = a->frame;
    enum AVSampleFormat out_fmt = fmt == 1 ? AV_SAMPLE_FMT_S32 : fmt == 2 ? AV_SAMPLE_FMT_FLT : AV_SAMPLE_FMT_S16;
    int dst_rate = a->out_rate > 0 ? a->out_rate : f->sample_rate;
    if (!a->swr || a->swr_in_fmt != f->format || a->swr_out_fmt != (int)out_fmt || a->swr_rate != f->sample_rate ||
        a->swr_out_rate != dst_rate || av_channel_layout_compare(&a->swr_layout, &f->ch_layout) != 0) {
        swr_free(&a->swr);
        av_channel_layout_uninit(&a->swr_layout);
        if (swr_alloc_set_opts2(&a->swr, &f->ch_layout, out_fmt, dst_rate, &f->ch_layout,
                                (enum AVSampleFormat)f->format, f->sample_rate, 0, NULL) < 0 ||
            swr_init(a->swr) < 0) {
            swr_free(&a->swr);
            return -2;
        }
        av_channel_layout_copy(&a->swr_layout, &f->ch_layout);
        a->swr_in_fmt = f->format;
        a->swr_out_fmt = (int)out_fmt;
        a->swr_rate = f->sample_rate;
        a->swr_out_rate = dst_rate;
    }
    int channels = f->ch_layout.nb_channels > 0 ? f->ch_layout.nb_channels : 1;
    int capacity = swr_get_out_samples(a->swr, f->nb_samples);
    if (capacity < 0) return -2;
    size_t need = (size_t)capacity * (size_t)channels * (size_t)av_get_bytes_per_sample(out_fmt);
    if (need > a->buf_size) {
        uint8_t *nb = realloc(a->buf, need);
        if (!nb) return -1;
        a->buf = nb;
        a->buf_size = need;
    }
    uint8_t *dst = a->buf;
    int n = swr_convert(a->swr, &dst, capacity, (const uint8_t **)f->extended_data, f->nb_samples);
    if (n < 0) return -2;
    out->channels = channels;
    out->sample_rate = dst_rate;
    out->nb_samples = n;
    out->bit_rate = a->ctx->bit_rate;
    out->profile = a->ctx->profile;
    return 1;
}

const uint8_t *psa_data(const PsaDecoder *a) { return a->buf; }
