// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Instruction opcodes.
//!
//! One flat enum covers every encoding; the instruction's [`Encoding`]
//! tells, e.g., a VOP2 `VCndmaskB32` from its VOP3 form. Variant names are
//! the canonical opcode identifiers (`SMovB32`); [`Opcode::mnemonic`] gives
//! the assembler spelling (`s_mov_b32`). Opcodes the decoder accepts without
//! naming them (unknown VOP3/VOP3B/VOP3P/MUBUF opcodes and the raw 0xFC top
//! byte) carry their raw opcode number and are reported by
//! [`Opcode::is_raw`].
//!
//! [`Encoding`]: crate::Encoding

use std::borrow::Cow;
use std::fmt;

macro_rules! opcodes {
    ($($v:ident),* $(,)?) => {
        /// A decoded instruction opcode.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum Opcode {
            $($v,)*
            /// FLAT / SCRATCH / GLOBAL memory operation.
            Flat(FlatSegment, FlatOp),
            /// VOP3 (VOP3A layout) opcode without a name.
            Vop3Raw(u16),
            /// VOP3B (scalar carry-out layout) opcode without a name.
            Vop3bRaw(u16),
            /// VOP3P opcode without a name.
            Vop3pRaw(u8),
            /// MUBUF opcode without a name.
            MubufRaw(u8),
            /// Two-dword instruction with top bits 0b111111 (word >> 24).
            Vop3pTopRaw(u8),
        }

        impl Opcode {
            /// Every named opcode except the FLAT family.
            pub const NAMED: &'static [Opcode] = &[$(Opcode::$v,)*];

            fn static_name(self) -> Option<&'static str> {
                match self {
                    $(Opcode::$v => Some(stringify!($v)),)*
                    _ => None,
                }
            }
        }
    };
}

opcodes! {
    // SOP1
    SMovB32, SMovB64, SCmovB64, SNotB32, SNotB64, SWqmB32, SWqmB64, SBrevB32,
    SBcnt1I32B32, SBcnt1I32B64, SFF1I32B32, SFF1I32B64, SFlbitI32B32, SFlbitI32B64,
    SBitset0B32, SBitset0B64, SBitset1B32, SBitset1B64, SGetpcB64, SSetpcB64, SSwappcB64,
    SAndSaveexecB64, SOrSaveexecB64, SXorSaveexecB64, SAndn2SaveexecB64, SOrn2SaveexecB64,
    SNandSaveexecB64, SNorSaveexecB64, SXnorSaveexecB64, SQuadmaskB32, SQuadmaskB64, SAbsI32,
    SAndn1SaveexecB64, SOrn1SaveexecB64, SBitreplicateB64B32,
    SAndSaveexecB32, SOrSaveexecB32, SXorSaveexecB32, SAndn2SaveexecB32, SOrn2SaveexecB32,
    SNandSaveexecB32, SNorSaveexecB32, SXnorSaveexecB32, SAndn1SaveexecB32, SOrn1SaveexecB32,
    // SOP2
    SAddU32, SSubU32, SAddI32, SSubI32, SAddcU32, SSubbU32, SMinI32, SMinU32, SMaxI32, SMaxU32,
    SCselectB32, SCselectB64, SAndB32, SAndB64, SOrB32, SOrB64, SXorB32, SXorB64,
    SAndn2B32, SAndn2B64, SOrn2B32, SOrn2B64, SNandB32, SNandB64, SNorB32, SNorB64,
    SXnorB32, SXnorB64, SLshlB32, SLshlB64, SLshrB32, SLshrB64, SAshrI32, SAshrI64,
    SBfmB32, SBfmB64, SMulI32, SBfeU32, SBfeI32, SBfeU64, SBfeI64, SAbsdiffI32,
    SLshl1AddU32, SLshl2AddU32, SLshl3AddU32, SLshl4AddU32,
    SPackLlB32B16, SPackLhB32B16, SPackHhB32B16, SMulHiU32, SMulHiI32, SAddF32, SSubF32,
    // SOPC
    SCmpEqI32, SCmpLgI32, SCmpGtI32, SCmpGeI32, SCmpLtI32, SCmpLeI32,
    SCmpEqU32, SCmpLgU32, SCmpGtU32, SCmpGeU32, SCmpLtU32, SCmpLeU32,
    SBitcmp0B32, SBitcmp1B32, SBitcmp0B64, SBitcmp1B64, SCmpEqU64, SCmpLgU64,
    // SOPP
    SNop, SEndpgm, SBranch, SCbranchScc0, SCbranchScc1, SCbranchVccz, SCbranchVccnz,
    SCbranchExecz, SCbranchExecnz, SBarrier, SWaitcnt, SSetprio, SSethalt, SSendmsg, STrap, STtraceData,
    SCbranchCdbgsys, SCbranchCdbguser, SCbranchCdbgsysOrUser, SCbranchCdbgsysAndUser,
    SCodeEnd, SInstPrefetch, SClause, SWaitcntDepctr,
    // SOPK
    SMovkI32, SCmpkEqI32, SCmpkLgI32, SCmpkGtI32, SCmpkGeI32, SCmpkLtI32, SCmpkLeI32,
    SCmpkEqU32, SCmpkLgU32, SCmpkGtU32, SCmpkGeU32, SCmpkLtU32, SCmpkLeU32,
    SAddkI32, SMulkI32, SSetregB32, SSubvectorLoopBegin, SSubvectorLoopEnd,
    // SMRD / SMEM
    SLoadDword, SLoadDwordx2, SLoadDwordx4, SLoadDwordx8, SLoadDwordx16,
    SBufferLoadDword, SBufferLoadDwordx2, SBufferLoadDwordx4, SBufferLoadDwordx8, SBufferLoadDwordx16,
    ScratchStoreDword, ScratchStoreDwordx2, ScratchStoreDwordx4,
    // VOP1
    VNop, VMovB32, VReadfirstlaneB32, VCvtF64I32, VCvtF32I32, VCvtF32U32, VCvtU32F32, VCvtI32F32,
    VCvtF16F32, VCvtF32F16, VCvtRpiI32F32, VCvtFlrI32F32, VCvtOffF32I4, VCvtF32F64,
    VCvtF32Ubyte0, VCvtF32Ubyte1, VCvtF32Ubyte2, VCvtF32Ubyte3, VCvtF64U32,
    VFractF32, VTruncF32, VCeilF32, VRndneF32, VFloorF32, VExpF32, VLogF32, VRcpF32, VRcpIflagF32,
    VRsqF32, VRcpF64, VRsqF64, VSqrtF32, VSqrtF64, VSinF32, VCosF32, VNotB32, VBfrevB32,
    VFfbhU32, VFfblB32, VFfbhI32, VMovreldB32, VMovrelsB32, VMovrelsdB32, VMovrelsd2B32,
    VCvtU16F16, VRcpF16, VSqrtF16, VRsqF16, VLogF16, VExpF16, VFloorF16, VCeilF16, VTruncF16,
    VRndneF16, VFractF16, VSinF16, VCosF16,
    // VOP2
    VCndmaskB32, VDot2cF32F16, VAddF32, VSubF32, VSubrevF32, VMulF32, VMulI32I24, VMulU32U24,
    VMulHiU32U24, VMinF32, VMaxF32, VMinI32, VMaxI32, VMinU32, VMaxU32, VLshrB32, VLshrrevB32,
    VAshrI32, VAshrrevI32, VLshlB32, VLshlrevB32, VAndB32, VOrB32, VXorB32, VXnorB32, VMacF32,
    VMadMkF32, VMadAkF32, VBcntU32B32, VMbcntLoU32B32, VMbcntHiU32B32, VAddI32, VSubI32,
    VSubrevI32, VAddcU32, VSubbU32, VSubbrevU32, VFmacF32, VFmaMkF32, VFmaAkF32,
    VCvtPkrtzF16F32, VCvtPkU16U32, VCvtPkI16I32, VAddF16, VSubF16, VSubrevF16, VMulF16,
    VFmaMkF16, VFmaAkF16, VMaxF16, VMinF16, VPkFmacF16,
    // VOPC (also VOP3 opcodes 0x000-0x0FF)
    VCmpFF32, VCmpLtF32, VCmpEqF32, VCmpLeF32, VCmpGtF32, VCmpLgF32, VCmpGeF32, VCmpOF32,
    VCmpUF32, VCmpNgeF32, VCmpNlgF32, VCmpNgtF32, VCmpNleF32, VCmpNeqF32, VCmpNltF32, VCmpTruF32,
    VCmpxFF32, VCmpxLtF32, VCmpxEqF32, VCmpxLeF32, VCmpxGtF32, VCmpxLgF32, VCmpxGeF32, VCmpxOF32,
    VCmpxUF32, VCmpxNgeF32, VCmpxNlgF32, VCmpxNgtF32, VCmpxNleF32, VCmpxNeqF32, VCmpxNltF32,
    VCmpxTruF32,
    VCmpFI32, VCmpLtI32, VCmpEqI32, VCmpLeI32, VCmpGtI32, VCmpNeI32, VCmpGeI32, VCmpTI32,
    VCmpClassF32, VCmpLtI16, VCmpEqI16, VCmpLeI16, VCmpGtI16, VCmpNeI16, VCmpGeI16,
    VCmpxClassF32, VCmpLtU16, VCmpEqU16, VCmpLeU16, VCmpGtU16, VCmpNeU16, VCmpGeU16,
    VCmpxLtI16, VCmpxEqI16, VCmpxLeI16, VCmpxGtI16, VCmpxNeI16, VCmpxGeI16,
    VCmpxLtU16, VCmpxEqU16, VCmpxLeU16, VCmpxGtU16, VCmpxNeU16, VCmpxGeU16,
    VCmpxFI32, VCmpxLtI32, VCmpxEqI32, VCmpxLeI32, VCmpxGtI32, VCmpxNeI32, VCmpxGeI32, VCmpxTI32,
    VCmpFI64, VCmpLtI64, VCmpEqI64, VCmpLeI64, VCmpGtI64, VCmpNeI64, VCmpGeI64, VCmpTI64,
    VCmpxFI64, VCmpxLtI64, VCmpxEqI64, VCmpxLeI64, VCmpxGtI64, VCmpxNeI64, VCmpxGeI64, VCmpxTI64,
    VCmpFU32, VCmpLtU32, VCmpEqU32, VCmpLeU32, VCmpGtU32, VCmpNeU32, VCmpGeU32, VCmpTU32,
    VCmpFF16, VCmpLtF16, VCmpEqF16, VCmpLeF16, VCmpGtF16, VCmpLgF16, VCmpGeF16, VCmpOF16,
    VCmpxFU32, VCmpxLtU32, VCmpxEqU32, VCmpxLeU32, VCmpxGtU32, VCmpxNeU32, VCmpxGeU32, VCmpxTU32,
    VCmpxFF16, VCmpxLtF16, VCmpxEqF16, VCmpxLeF16, VCmpxGtF16, VCmpxLgF16, VCmpxGeF16, VCmpxOF16,
    VCmpFU64, VCmpLtU64, VCmpEqU64, VCmpLeU64, VCmpGtU64, VCmpNeU64, VCmpGeU64, VCmpTU64,
    VCmpUF16, VCmpNgeF16, VCmpNlgF16, VCmpNgtF16, VCmpNleF16, VCmpNeqF16, VCmpNltF16, VCmpTruF16,
    VCmpxFU64, VCmpxLtU64, VCmpxEqU64, VCmpxLeU64, VCmpxGtU64, VCmpxNeU64, VCmpxGeU64, VCmpxTU64,
    VCmpxUF16, VCmpxNgeF16, VCmpxNlgF16, VCmpxNgtF16, VCmpxNleF16, VCmpxNeqF16, VCmpxNltF16,
    VCmpxTruF16,
    // VOP3 only
    VMadF32, VMadI32I24, VMadU32U24, VCubeidF32, VCubescF32, VCubetcF32, VCubemaF32, VBfiB32,
    VAlignbitB32, VAlignbyteB32, VFmaF32, VFmaF64, VMin3F32, VMin3I32, VMin3U32, VMax3F32,
    VMax3I32, VMax3U32, VMed3F32, VMed3I32, VMed3U32, VSadU8, VSadHiU8, VSadU16, VSadU32,
    VCvtPkU8F32, VBfeU32, VBfeI32, VMulF64, VMulLoU32, VMulHiU32, VMulLoI32, VMulHiI32,
    VAddNcU16, VFmaF16, VMin3F16, VMax3F16, VMed3F16, VReadlaneB32, VWritelaneB32, VLdexpF32,
    VBfmB32, VCvtPknormI16F32, VCvtPknormU16F32, VLshlrevB64, VLshrrevB64, VMadU32U16,
    VLshlAddU32, VXadU32, VAddLshlU32, VAdd3U32, VLshlOrU32, VXor3B32, VAndOrB32, VOr3U32,
    VPermlane16B32, VPermlanex16B32, VSubNcU16, VLshrrevB16, VAshrrevI16, VMaxU16, VMaxI16,
    VMinU16, VMinI16, VAddNcI16, VSubNcI16, VLshlrevB16,
    // VOP3B
    VAddCoCiU32, VSubCoCiU32, VSubrevCoCiU32, VAddCoU32, VSubCoU32, VSubrevCoU32, VMadU64U32,
    // VOP3P
    VPkAddI16, VPkSubI16, VPkFmaF16, VPkAddF16, VPkMulF16, VPkMinF16, VPkMaxF16,
    VFmaMixF32, VFmaMixloF16, VFmaMixhiF16,
    // DS
    DsAddU32, DsSubU32, DsIncU32, DsDecU32, DsMinI32, DsMaxI32, DsMinU32, DsMaxU32, DsAndB32,
    DsOrB32, DsXorB32, DsWriteB32, DsWrite2B32, DsWrite2St64B32, DsCmpstB32, DsMinF32, DsMaxF32,
    DsAddRtnU32, DsSubRtnU32, DsIncRtnU32, DsDecRtnU32, DsMinRtnI32, DsMaxRtnI32, DsMinRtnU32,
    DsMaxRtnU32, DsAndRtnB32, DsOrRtnB32, DsXorRtnB32, DsWrxchgRtnB32, DsCmpstRtnB32,
    DsSwizzleB32, DsReadB32, DsRead2B32, DsRead2St64B32, DsReadI8, DsConsume, DsAppend, DsOrderedCount,
    DsAddU64, DsOrB64, DsWriteB64, DsWrite2B64, DsWrite2St64B64, DsReadB64, DsRead2B64,
    DsWriteAddtidB32, DsReadAddtidB32, DsBpermuteB32, DsWriteB96, DsWriteB128, DsReadB96,
    DsReadB128,
    // Sub-dword LDS access (gfx10 DS 0x1E/0x1F/0x3A-0x3C).
    DsWriteB8, DsWriteB16, DsReadU8, DsReadI16, DsReadU16,
    // MTBUF
    TBufferLoadFormatX, TBufferLoadFormatXy, TBufferLoadFormatXyz, TBufferLoadFormatXyzw,
    TBufferStoreFormatX, TBufferStoreFormatXy, TBufferStoreFormatXyz, TBufferStoreFormatXyzw,
    TBufferLoadFormatD16X, TBufferLoadFormatD16Xy, TBufferLoadFormatD16Xyz, TBufferLoadFormatD16Xyzw,
    TBufferStoreFormatD16X, TBufferStoreFormatD16Xy, TBufferStoreFormatD16Xyz,
    TBufferStoreFormatD16Xyzw,
    // MUBUF
    BufferLoadFormatX, BufferLoadFormatXy, BufferLoadFormatXyz, BufferLoadFormatXyzw,
    BufferStoreFormatX, BufferStoreFormatXy, BufferStoreFormatXyz, BufferStoreFormatXyzw,
    BufferLoadUbyte, BufferLoadSbyte, BufferLoadUshort, BufferLoadSshort, BufferLoadDword,
    BufferLoadDwordx2, BufferLoadDwordx4, BufferLoadDwordx3, BufferStoreByte, BufferStoreByteD16Hi,
    BufferStoreShort, BufferStoreShortD16Hi, BufferStoreDword, BufferStoreDwordx2,
    BufferStoreDwordx4, BufferStoreDwordx3, BufferLoadUbyteD16, BufferLoadUbyteD16Hi,
    BufferLoadSbyteD16, BufferLoadSbyteD16Hi, BufferLoadShortD16, BufferLoadShortD16Hi,
    BufferLoadFormatD16X, BufferLoadFormatD16Xy, BufferLoadFormatD16Xyz, BufferLoadFormatD16Xyzw,
    BufferStoreFormatD16X, BufferStoreFormatD16Xy, BufferStoreFormatD16Xyz, BufferStoreFormatD16Xyzw,
    BufferLoadFormatD16HiX, BufferStoreFormatD16HiX,
    BufferAtomicSwap, BufferAtomicCmpswap, BufferAtomicAdd, BufferAtomicSub, BufferAtomicSmin,
    BufferAtomicUmin, BufferAtomicSmax, BufferAtomicUmax, BufferAtomicAnd, BufferAtomicOr,
    BufferAtomicXor, BufferAtomicInc, BufferAtomicDec, BufferWbinvl1, BufferAtomicFmin,
    BufferAtomicFmax, BufferAtomicSwapX2, BufferAtomicOrX2,
    // MIMG
    ImageLoad, ImageLoadMip, ImageStore, ImageStoreMip, ImageGetResinfo, ImageAtomicSwap,
    ImageAtomicCmpswap, ImageAtomicAdd, ImageAtomicSub, ImageAtomicSmin, ImageAtomicUmin,
    ImageAtomicSmax, ImageAtomicUmax, ImageAtomicAnd, ImageAtomicOr, ImageAtomicXor,
    ImageAtomicInc, ImageAtomicDec, ImageAtomicFcmpswap, ImageAtomicFmin, ImageAtomicFmax,
    ImageSample, ImageSampleCl, ImageSampleD, ImageSampleDCl, ImageSampleL, ImageSampleB,
    ImageSampleBCl, ImageSampleLz, ImageSampleC, ImageSampleCCl, ImageSampleCD, ImageSampleCDCl,
    ImageSampleCL, ImageSampleCB, ImageSampleCBCl, ImageSampleCLz, ImageSampleO, ImageSampleClO,
    ImageSampleDO, ImageSampleDClO, ImageSampleLO, ImageSampleBO, ImageSampleBClO, ImageSampleLzO,
    ImageSampleCO, ImageSampleCClO, ImageSampleCDO, ImageSampleCDClO, ImageSampleCLO,
    ImageSampleCBO, ImageSampleCBClO, ImageSampleCLzO, ImageGather4, ImageGather4Lz,
    ImageGather4C, ImageGather4CBCl, ImageGather4CLz, ImageGather4LzO, ImageGather4CO,
    ImageGather4CLzO, ImageGetLod, ImageGather4H, ImageGather4Cl, ImageGather4L, ImageGather4B,
    ImageGather4BCl, ImageGather4CCl, ImageGather4CL, ImageGather4CB, ImageGather4O, ImageGather4ClO,
    ImageGather4LO, ImageGather4BO, ImageGather4BClO, ImageGather4CClO, ImageGather4CLO, ImageGather4CBO,
    ImageGather4CBClO, ImageSampleCd, ImageSampleCdCl, ImageSampleCCd,
    ImageSampleCCdCl, ImageSampleCdO, ImageSampleCdClO, ImageSampleCCdO, ImageSampleCCdClO,
    ImageSampleA, ImageSampleClA, ImageSampleBA, ImageSampleBClA, ImageSampleCA, ImageSampleCClA,
    ImageSampleCBA, ImageSampleCBClA, ImageSampleAO, ImageSampleClAO, ImageSampleBAO,
    ImageSampleBClAO, ImageSampleCAO, ImageSampleCClAO, ImageSampleCBAO, ImageSampleCBClAO,
    // MIMG samples with 16-bit gradients (G16)
    ImageSampleDG16, ImageSampleDClG16, ImageSampleCDG16, ImageSampleCDClG16, ImageSampleDOG16,
    ImageSampleDClOG16, ImageSampleCDOG16, ImageSampleCDClOG16, ImageSampleCdG16, ImageSampleCdClG16,
    ImageSampleCCdG16, ImageSampleCCdClG16, ImageSampleCdOG16, ImageSampleCdClOG16, ImageSampleCCdOG16,
    ImageSampleCCdClOG16,
    ImageBvhIntersectRay, ImageBvh64IntersectRay, ImageMsaaLoad,
    // PS5 compare gathers with the adjusted sampler (_a).
    ImageGather4CA, ImageGather4CAO,
    // MIMG raw-texel (PCK), multi-element (BY2 / BY4, PCK2 / PCK4) forms.
    ImageLoadPck, ImageLoadPckSgn, ImageLoadMipPck, ImageLoadMipPckSgn, ImageStorePck, ImageStoreMipPck,
    ImageLoadBy2, ImageLoadBy4, ImageLoadMipBy2, ImageLoadMipBy4, ImageStoreBy2, ImageStoreBy4,
    ImageStoreMipBy2, ImageStoreMipBy4, ImageLoadPck2, ImageLoadPck4, ImageLoadMipPck2, ImageLoadMipPck4,
    ImageStorePck2, ImageStorePck4, ImageStoreMipPck2, ImageStoreMipPck4, ImageGather4hPck, ImageGather8hPck,
    // Less common forms decoded for completeness.
    SBrevB64, SBcnt0I32B32, SBcnt0I32B64, SFF0I32B32, SFF0I32B64, SFlbitI32, SFlbitI32I64,
    SSextI32I8, SSextI32I16, SMovrelsB32, SMovrelsB64, SMovreldB32, SMovreldB64, SMovrelsd2B32,
    SCmovB32, SAndn1WrexecB64, SAndn2WrexecB64, SAndn1WrexecB32, SAndn2WrexecB32,
    SCmovkI32, SGetregB32, SSetregImm32B32, SVersion, SSleep, SWakeup, SIcacheInv, SIncperflevel,
    SDecperflevel, SRoundMode, SDenormMode, STtracedataImm, SEndpgmSaved,
    SEndpgmOrderedPsDone, SSetkill, SWaitIdle, SGl1Inv, SDcacheInv, SDcacheWb, SMemtime, SMemrealtime,
    SAtcProbe, SAtcProbeBuffer, SDcacheDiscard, SDcacheDiscardX2, SGetWaveidInWorkgroup,
    BufferGl0Inv, BufferGl1Inv, DsNop, DsRead2St64B64,
    VCvtI32F64, VCvtF64F32, VCvtU32F64, VTruncF64, VCeilF64, VRndneF64, VFloorF64, VPipeflush,
    VFrexpExpI32F64, VFrexpMantF64, VFractF64, VFrexpExpI32F32, VFrexpMantF32, VClrexcp,
    VCvtF16U16, VCvtF16I16, VCvtI16F16, VFrexpMantF16, VFrexpExpI16F16, VSatPkU8I16,
    VCvtNormI16F16, VCvtNormU16F16, VSwapB32, VSwaprelB32,
    VMacLegacyF32, VMulLegacyF32, VMulHiI32I24, VDot4cI32I8, VFmacF16, VLdexpF16,
    VMadLegacyF32, VLerpU8, VMullitF32, VDivFixupF32, VDivFixupF64, VAddF64, VMinF64, VMaxF64,
    VLdexpF64, VDivScaleF32, VDivScaleF64, VDivFmasF32, VDivFmasF64, VMsadU8, VQsadPkU16U8,
    VMqsadPkU16U8, VTrigPreopF64, VMqsadU32U8, VMadI64I32, VAshrrevI64, VMulLoU16, VPackB32F16,
    VCvtPknormI16F16, VCvtPknormU16F16, VMadU16, VPermB32, VMin3I16, VMin3U16, VMax3I16,
    VMax3U16, VMed3I16, VMed3U16, VMadI16, VDivFixupF16, VMadI32I16, VSubNcI32, VAddNcI32,
    VPkMadI16, VPkMulLoU16, VPkLshlrevB16, VPkLshrrevB16, VPkAshrrevI16, VPkMaxI16, VPkMinI16,
    VPkMadU16, VPkAddU16, VPkSubU16, VPkMaxU16, VPkMinU16, VDot2F32F16, VDot2I32I16,
    VDot2U32U16, VDot4I32I8, VDot4U32U8, VDot8I32I4, VDot8U32U4,
    VCmpFF64, VCmpLtF64, VCmpEqF64, VCmpLeF64, VCmpGtF64, VCmpLgF64, VCmpGeF64, VCmpOF64,
    VCmpUF64, VCmpNgeF64, VCmpNlgF64, VCmpNgtF64, VCmpNleF64, VCmpNeqF64, VCmpNltF64, VCmpTruF64,
    VCmpxFF64, VCmpxLtF64, VCmpxEqF64, VCmpxLeF64, VCmpxGtF64, VCmpxLgF64, VCmpxGeF64, VCmpxOF64,
    VCmpxUF64, VCmpxNgeF64, VCmpxNlgF64, VCmpxNgtF64, VCmpxNleF64, VCmpxNeqF64, VCmpxNltF64,
    VCmpxTruF64, VCmpClassF64, VCmpxClassF64, VCmpClassF16, VCmpxClassF16,
    DsRsubU32, DsMskorB32, DsCmpstF32, DsAddF32, DsRsubRtnU32, DsMskorRtnB32, DsWrxchg2RtnB32,
    DsWrxchg2St64RtnB32, DsCmpstRtnF32, DsMinRtnF32, DsMaxRtnF32, DsWrapRtnB32, DsAddRtnF32,
    DsAddSrc2U32, DsSubSrc2U32, DsRsubSrc2U32, DsIncSrc2U32, DsDecSrc2U32, DsMinSrc2I32, DsMaxSrc2I32, DsMinSrc2U32,
    DsMaxSrc2U32, DsAndSrc2B32, DsOrSrc2B32, DsXorSrc2B32, DsWriteSrc2B32, DsMinSrc2F32, DsMaxSrc2F32, DsAddSrc2F32,
    DsPermuteB32, DsSubU64, DsRsubU64, DsIncU64, DsDecU64, DsMinI64, DsMaxI64, DsMinU64,
    DsMaxU64, DsAndB64, DsXorB64, DsMskorB64, DsCmpstB64, DsCmpstF64, DsMinF64, DsMaxF64,
    DsAddRtnU64, DsSubRtnU64, DsRsubRtnU64, DsIncRtnU64, DsDecRtnU64, DsMinRtnI64, DsMaxRtnI64,
    DsMinRtnU64, DsMaxRtnU64, DsAndRtnB64, DsOrRtnB64, DsXorRtnB64, DsMskorRtnB64,
    DsWrxchgRtnB64, DsWrxchg2RtnB64, DsWrxchg2St64RtnB64, DsCmpstRtnB64, DsCmpstRtnF64,
    DsMinRtnF64, DsMaxRtnF64, DsReadI8D16, DsReadU8D16, DsReadI8D16Hi, DsReadU8D16Hi,
    DsReadU16D16, DsReadU16D16Hi, DsWriteB8D16Hi, DsWriteB16D16Hi,
    BufferAtomicSubX2, BufferAtomicCmpswapX2, BufferAtomicAddX2, BufferAtomicSminX2,
    BufferAtomicUminX2, BufferAtomicSmaxX2, BufferAtomicUmaxX2, BufferAtomicAndX2,
    BufferAtomicXorX2, BufferAtomicIncX2, BufferAtomicDecX2, BufferAtomicCsub,
    BufferAtomicFcmpswap, BufferAtomicFcmpswapX2, BufferAtomicFminX2, BufferAtomicFmaxX2,
    // Decoded so that they are refused by name (trap return, halting
    // message, scalar scratch, global wave sync) or translated (the
    // conditional exchange) or refused for want of call support.
    SRfeB64, SSendmsghalt, SCallB64, SScratchLoadDword, SScratchLoadDwordx2, SScratchLoadDwordx4,
    DsGwsSemaReleaseAll, DsGwsInit, DsGwsSemaV, DsGwsSemaBr, DsGwsSemaP, DsGwsBarrier, DsCondxchg32RtnB64,
    // VINTRP
    VInterpP1F32, VInterpP2F32, VInterpMovF32,
    // VOP3 16-bit interpolation
    VInterpP1llF16, VInterpP1lvF16, VInterpP2F16,
    // EXP
    Exp,
}

/// Address space of a FLAT-family instruction (word bits 15:14).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FlatSegment {
    Flat,
    Scratch,
    Global,
}

impl FlatSegment {
    pub fn name(self) -> &'static str {
        match self {
            FlatSegment::Flat => "Flat",
            FlatSegment::Scratch => "Scratch",
            FlatSegment::Global => "Global",
        }
    }
}

/// Operation of a FLAT-family instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FlatOp {
    LoadUbyte,
    LoadSbyte,
    LoadUshort,
    LoadSshort,
    LoadDword,
    LoadDwordx2,
    LoadDwordx4,
    LoadDwordx3,
    StoreByte,
    StoreByteD16Hi,
    StoreShort,
    StoreShortD16Hi,
    StoreDword,
    StoreDwordx2,
    StoreDwordx4,
    StoreDwordx3,
    LoadUbyteD16,
    LoadUbyteD16Hi,
    LoadSbyteD16,
    LoadSbyteD16Hi,
    LoadShortD16,
    LoadShortD16Hi,
    AtomicAdd,
    AtomicUMax,
    AtomicSwap,
    AtomicCmpswap,
    AtomicSub,
    AtomicCsub,
    AtomicSMin,
    AtomicUMin,
    AtomicSMax,
    AtomicAnd,
    AtomicOr,
    AtomicXor,
    AtomicInc,
    AtomicDec,
    AtomicFcmpswap,
    AtomicFMin,
    AtomicFMax,
    AtomicSwapX2,
    AtomicCmpswapX2,
    AtomicAddX2,
    AtomicSubX2,
    AtomicSMinX2,
    AtomicUMinX2,
    AtomicSMaxX2,
    AtomicUMaxX2,
    AtomicAndX2,
    AtomicOrX2,
    AtomicXorX2,
    AtomicIncX2,
    AtomicDecX2,
    AtomicFcmpswapX2,
    AtomicFMinX2,
    AtomicFMaxX2,
    LoadDwordAddtid,
    StoreDwordAddtid,
}

impl FlatOp {
    pub const ALL: [FlatOp; 57] = [
        FlatOp::LoadUbyte,
        FlatOp::LoadSbyte,
        FlatOp::LoadUshort,
        FlatOp::LoadSshort,
        FlatOp::LoadDword,
        FlatOp::LoadDwordx2,
        FlatOp::LoadDwordx4,
        FlatOp::LoadDwordx3,
        FlatOp::StoreByte,
        FlatOp::StoreByteD16Hi,
        FlatOp::StoreShort,
        FlatOp::StoreShortD16Hi,
        FlatOp::StoreDword,
        FlatOp::StoreDwordx2,
        FlatOp::StoreDwordx4,
        FlatOp::StoreDwordx3,
        FlatOp::LoadUbyteD16,
        FlatOp::LoadUbyteD16Hi,
        FlatOp::LoadSbyteD16,
        FlatOp::LoadSbyteD16Hi,
        FlatOp::LoadShortD16,
        FlatOp::LoadShortD16Hi,
        FlatOp::AtomicAdd,
        FlatOp::AtomicUMax,
        FlatOp::AtomicSwap,
        FlatOp::AtomicCmpswap,
        FlatOp::AtomicSub,
        FlatOp::AtomicCsub,
        FlatOp::AtomicSMin,
        FlatOp::AtomicUMin,
        FlatOp::AtomicSMax,
        FlatOp::AtomicAnd,
        FlatOp::AtomicOr,
        FlatOp::AtomicXor,
        FlatOp::AtomicInc,
        FlatOp::AtomicDec,
        FlatOp::AtomicFcmpswap,
        FlatOp::AtomicFMin,
        FlatOp::AtomicFMax,
        FlatOp::AtomicSwapX2,
        FlatOp::AtomicCmpswapX2,
        FlatOp::AtomicAddX2,
        FlatOp::AtomicSubX2,
        FlatOp::AtomicSMinX2,
        FlatOp::AtomicUMinX2,
        FlatOp::AtomicSMaxX2,
        FlatOp::AtomicUMaxX2,
        FlatOp::AtomicAndX2,
        FlatOp::AtomicOrX2,
        FlatOp::AtomicXorX2,
        FlatOp::AtomicIncX2,
        FlatOp::AtomicDecX2,
        FlatOp::AtomicFcmpswapX2,
        FlatOp::AtomicFMinX2,
        FlatOp::AtomicFMaxX2,
        FlatOp::LoadDwordAddtid,
        FlatOp::StoreDwordAddtid,
    ];

    pub fn name(self) -> &'static str {
        match self {
            FlatOp::LoadUbyte => "LoadUbyte",
            FlatOp::LoadSbyte => "LoadSbyte",
            FlatOp::LoadUshort => "LoadUshort",
            FlatOp::LoadSshort => "LoadSshort",
            FlatOp::LoadDword => "LoadDword",
            FlatOp::LoadDwordx2 => "LoadDwordx2",
            FlatOp::LoadDwordx4 => "LoadDwordx4",
            FlatOp::LoadDwordx3 => "LoadDwordx3",
            FlatOp::StoreByte => "StoreByte",
            FlatOp::StoreByteD16Hi => "StoreByteD16Hi",
            FlatOp::StoreShort => "StoreShort",
            FlatOp::StoreShortD16Hi => "StoreShortD16Hi",
            FlatOp::StoreDword => "StoreDword",
            FlatOp::StoreDwordx2 => "StoreDwordx2",
            FlatOp::StoreDwordx4 => "StoreDwordx4",
            FlatOp::StoreDwordx3 => "StoreDwordx3",
            FlatOp::LoadUbyteD16 => "LoadUbyteD16",
            FlatOp::LoadUbyteD16Hi => "LoadUbyteD16Hi",
            FlatOp::LoadSbyteD16 => "LoadSbyteD16",
            FlatOp::LoadSbyteD16Hi => "LoadSbyteD16Hi",
            FlatOp::LoadShortD16 => "LoadShortD16",
            FlatOp::LoadShortD16Hi => "LoadShortD16Hi",
            FlatOp::AtomicAdd => "AtomicAdd",
            FlatOp::AtomicUMax => "AtomicUMax",
            FlatOp::AtomicSwap => "AtomicSwap",
            FlatOp::AtomicCmpswap => "AtomicCmpswap",
            FlatOp::AtomicSub => "AtomicSub",
            FlatOp::AtomicCsub => "AtomicCsub",
            FlatOp::AtomicSMin => "AtomicSMin",
            FlatOp::AtomicUMin => "AtomicUMin",
            FlatOp::AtomicSMax => "AtomicSMax",
            FlatOp::AtomicAnd => "AtomicAnd",
            FlatOp::AtomicOr => "AtomicOr",
            FlatOp::AtomicXor => "AtomicXor",
            FlatOp::AtomicInc => "AtomicInc",
            FlatOp::AtomicDec => "AtomicDec",
            FlatOp::AtomicFcmpswap => "AtomicFcmpswap",
            FlatOp::AtomicFMin => "AtomicFMin",
            FlatOp::AtomicFMax => "AtomicFMax",
            FlatOp::AtomicSwapX2 => "AtomicSwapX2",
            FlatOp::AtomicCmpswapX2 => "AtomicCmpswapX2",
            FlatOp::AtomicAddX2 => "AtomicAddX2",
            FlatOp::AtomicSubX2 => "AtomicSubX2",
            FlatOp::AtomicSMinX2 => "AtomicSMinX2",
            FlatOp::AtomicUMinX2 => "AtomicUMinX2",
            FlatOp::AtomicSMaxX2 => "AtomicSMaxX2",
            FlatOp::AtomicUMaxX2 => "AtomicUMaxX2",
            FlatOp::AtomicAndX2 => "AtomicAndX2",
            FlatOp::AtomicOrX2 => "AtomicOrX2",
            FlatOp::AtomicXorX2 => "AtomicXorX2",
            FlatOp::AtomicIncX2 => "AtomicIncX2",
            FlatOp::AtomicDecX2 => "AtomicDecX2",
            FlatOp::AtomicFcmpswapX2 => "AtomicFcmpswapX2",
            FlatOp::AtomicFMinX2 => "AtomicFMinX2",
            FlatOp::AtomicFMaxX2 => "AtomicFMaxX2",
            FlatOp::LoadDwordAddtid => "LoadDwordAddtid",
            FlatOp::StoreDwordAddtid => "StoreDwordAddtid",

        }
    }

    pub fn is_load(self) -> bool {
        self.name().starts_with("Load")
    }

    pub fn is_store(self) -> bool {
        self.name().starts_with("Store")
    }

    pub fn is_atomic(self) -> bool {
        self.name().starts_with("Atomic")
    }

    /// Partial (16-bit) load that preserves the other destination half.
    pub fn is_d16_load(self) -> bool {
        self.is_load() && self.name().contains("D16")
    }

    /// Dwords an atomic returns (glc).
    pub fn return_dwords(self) -> u32 {
        if self.name().ends_with("X2") {
            2
        } else {
            1
        }
    }

    /// Dwords moved per lane (sub-dword accesses count as one).
    pub fn dword_count(self) -> u32 {
        match self {
            FlatOp::LoadDwordx2 | FlatOp::StoreDwordx2 => 2,
            FlatOp::LoadDwordx3 | FlatOp::StoreDwordx3 => 3,
            FlatOp::LoadDwordx4 | FlatOp::StoreDwordx4 => 4,
            FlatOp::AtomicCmpswapX2 | FlatOp::AtomicFcmpswapX2 => 4,
            FlatOp::AtomicCmpswap | FlatOp::AtomicFcmpswap => 2,
            op if op.name().ends_with("X2") && op.is_atomic() => 2,
            _ => 1,
        }
    }
}

impl Opcode {
    /// Canonical identifier, e.g. `SMovB32`, `GlobalLoadDword`, `Vop3Raw1A5`.
    pub fn name(self) -> Cow<'static, str> {
        if let Some(n) = self.static_name() {
            return Cow::Borrowed(n);
        }
        Cow::Owned(match self {
            Opcode::Flat(s, op) => format!("{}{}", s.name(), op.name()),
            Opcode::Vop3Raw(op) => format!("Vop3Raw{op:03X}"),
            Opcode::Vop3bRaw(op) => format!("Vop3bRaw{op:03X}"),
            Opcode::Vop3pRaw(op) => format!("Vop3pRaw{op:02X}"),
            Opcode::MubufRaw(op) => format!("MubufRaw{op:02X}"),
            Opcode::Vop3pTopRaw(b) => format!("Vop3pRaw{b:02X}"),
            _ => unreachable!(),
        })
    }

    /// Assembler mnemonic, e.g. `s_mov_b32`, `global_load_dword`.
    pub fn mnemonic(self) -> String {
        match self {
            Opcode::Vop3Raw(op) => return format!("v_op3_raw_{op:03x}"),
            Opcode::Vop3bRaw(op) => return format!("v_op3b_raw_{op:03x}"),
            Opcode::Vop3pRaw(op) => return format!("v_op3p_raw_{op:02x}"),
            Opcode::MubufRaw(op) => return format!("buffer_raw_{op:02x}"),
            Opcode::Vop3pTopRaw(b) => return format!("raw_top_{b:02x}"),
            Opcode::SFF1I32B32 => return "s_ff1_i32_b32".into(),
            Opcode::SFF1I32B64 => return "s_ff1_i32_b64".into(),
            Opcode::SFF0I32B32 => return "s_ff0_i32_b32".into(),
            Opcode::SFF0I32B64 => return "s_ff0_i32_b64".into(),
            Opcode::VMadMkF32 => return "v_madmk_f32".into(),
            Opcode::VMadAkF32 => return "v_madak_f32".into(),
            Opcode::VFmaMkF32 => return "v_fmamk_f32".into(),
            Opcode::VFmaAkF32 => return "v_fmaak_f32".into(),
            Opcode::VFmaMkF16 => return "v_fmamk_f16".into(),
            Opcode::VFmaAkF16 => return "v_fmaak_f16".into(),
            _ => {}
        }
        let name = self.name();
        let (prefix, rest) = match name.strip_prefix("TBuffer") {
            Some(rest) => ("tbuffer", rest),
            None => ("", &name[..]),
        };
        let mut out = String::from(prefix);
        for (i, c) in rest.chars().enumerate() {
            if c.is_ascii_uppercase() {
                if i != 0 || !out.is_empty() {
                    out.push('_');
                }
                out.push(c.to_ascii_lowercase());
            } else {
                out.push(c);
            }
        }
        out
    }

    /// Accepted by the decoder without a name (no semantics known).
    pub fn is_raw(self) -> bool {
        matches!(
            self,
            Opcode::Vop3Raw(_) | Opcode::Vop3bRaw(_) | Opcode::Vop3pRaw(_) | Opcode::MubufRaw(_) | Opcode::Vop3pTopRaw(_)
        )
    }

    // ---- classification used by the analyses -------------------------------

    /// V_CMP* and V_CMPX* (including CLASS).
    pub fn is_vector_compare(self) -> bool {
        self.name().starts_with("VCmp")
    }

    /// V_CMPX*: the compare result goes to EXEC (only, on gfx10).
    pub fn is_vector_compare_exec(self) -> bool {
        self.name().starts_with("VCmpx")
    }

    /// S_*_SAVEEXEC_B64 / _B32.
    pub fn is_saveexec(self) -> bool {
        self.name().contains("Saveexec")
    }

    pub fn is_saveexec_b64(self) -> bool {
        self.name().ends_with("SaveexecB64")
    }

    pub fn is_saveexec_b32(self) -> bool {
        self.name().ends_with("SaveexecB32")
    }

    /// Scalar ALU op whose destination is a 64-bit register pair.
    pub fn is_wide_scalar(self) -> bool {
        let n = self.name();
        n.starts_with('S') && (n.ends_with("B64") || n.ends_with("U64") || n.ends_with("I64"))
    }

    /// Vector carry-out ops that write VCC (VOP2) or an SGPR pair (VOP3B).
    pub fn is_vector_carry(self) -> bool {
        matches!(
            self,
            Opcode::VAddCoCiU32
                | Opcode::VSubCoCiU32
                | Opcode::VSubrevCoCiU32
                | Opcode::VAddCoU32
                | Opcode::VSubCoU32
                | Opcode::VSubrevCoU32
                | Opcode::VAddcU32
                | Opcode::VSubbU32
                | Opcode::VSubbrevU32
        )
    }

    /// Carry-in readers (third source is the carry mask).
    pub fn reads_vector_carry_in(self) -> bool {
        matches!(
            self,
            Opcode::VAddcU32
                | Opcode::VAddCoCiU32
                | Opcode::VSubbU32
                | Opcode::VSubCoCiU32
                | Opcode::VSubbrevU32
                | Opcode::VSubrevCoCiU32
        )
    }

    /// Carry-out writers (lane mask destination).
    pub fn writes_vector_carry_out(self) -> bool {
        self.reads_vector_carry_in()
            || matches!(self, Opcode::VAddCoU32 | Opcode::VSubCoU32 | Opcode::VSubrevCoU32 | Opcode::VMadU64U32)
    }

    /// V_MOVRELS/V_MOVRELD/V_MOVRELSD (M0-relative VGPR moves).
    pub fn is_bounded_relative_move(self) -> bool {
        matches!(self, Opcode::VMovrelsB32 | Opcode::VMovreldB32 | Opcode::VMovrelsdB32)
    }

    /// Ends the wave: S_ENDPGM and its saved / ordered-PS-done forms, and
    /// S_SETKILL (measured to end the wave whatever its operand).
    pub fn is_program_end(self) -> bool {
        matches!(self, Opcode::SEndpgm | Opcode::SEndpgmSaved | Opcode::SEndpgmOrderedPsDone | Opcode::SSetkill)
    }

    /// Computed jumps (S_SETPC_B64, S_SWAPPC_B64).
    pub fn is_computed_jump(self) -> bool {
        matches!(self, Opcode::SSetpcB64 | Opcode::SSwappcB64)
    }

    /// S_CBRANCH_CDBG*: taken only under an attached GPU debugger.
    pub fn is_debugger_branch(self) -> bool {
        matches!(
            self,
            Opcode::SCbranchCdbgsys | Opcode::SCbranchCdbguser | Opcode::SCbranchCdbgsysOrUser | Opcode::SCbranchCdbgsysAndUser
        )
    }

    pub fn is_image(self) -> bool {
        self.name().starts_with("Image")
    }

    pub fn is_image_load(self) -> bool {
        matches!(self, Opcode::ImageLoad | Opcode::ImageLoadMip)
    }

    pub fn is_image_store(self) -> bool {
        self.name().starts_with("ImageStore")
    }

    /// Writes through an image (store or atomic).
    pub fn is_storage_image(self) -> bool {
        let n = self.name();
        n.starts_with("ImageStore") || n.starts_with("ImageAtomic")
    }

    pub fn is_ray_intersect(self) -> bool {
        matches!(self, Opcode::ImageBvhIntersectRay | Opcode::ImageBvh64IntersectRay)
    }

    /// LDS atomics (with or without return).
    pub fn is_data_share_atomic(self) -> bool {
        self.ds_atomic_shape().is_some()
    }

    /// Shape of an LDS atomic: `(64-bit data, data operands (1 or 2),
    /// returns the old value)`.
    pub fn ds_atomic_shape(self) -> Option<(bool, u32, bool)> {
        use Opcode::*;
        Some(match self {
            DsAddU32 | DsSubU32 | DsRsubU32 | DsIncU32 | DsDecU32 | DsMinI32 | DsMaxI32 | DsMinU32 | DsMaxU32
            | DsAndB32 | DsOrB32 | DsXorB32 | DsAddF32 => (false, 1, false),
            DsMskorB32 | DsCmpstB32 | DsCmpstF32 | DsMinF32 | DsMaxF32 => (false, 2, false),
            DsAddRtnU32 | DsSubRtnU32 | DsRsubRtnU32 | DsIncRtnU32 | DsDecRtnU32 | DsMinRtnI32 | DsMaxRtnI32
            | DsMinRtnU32 | DsMaxRtnU32 | DsAndRtnB32 | DsOrRtnB32 | DsXorRtnB32 | DsWrxchgRtnB32
            | DsAddRtnF32 => (false, 1, true),
            DsMskorRtnB32 | DsCmpstRtnB32 | DsCmpstRtnF32 | DsMinRtnF32 | DsMaxRtnF32 | DsWrapRtnB32 | DsWrxchg2RtnB32
            | DsWrxchg2St64RtnB32 => (false, 2, true),
            DsAddU64 | DsSubU64 | DsRsubU64 | DsIncU64 | DsDecU64 | DsMinI64 | DsMaxI64 | DsMinU64 | DsMaxU64
            | DsAndB64 | DsOrB64 | DsXorB64 => (true, 1, false),
            DsMskorB64 | DsCmpstB64 | DsCmpstF64 | DsMinF64 | DsMaxF64 => (true, 2, false),
            DsAddRtnU64 | DsSubRtnU64 | DsRsubRtnU64 | DsIncRtnU64 | DsDecRtnU64 | DsMinRtnI64 | DsMaxRtnI64
            | DsMinRtnU64 | DsMaxRtnU64 | DsAndRtnB64 | DsOrRtnB64 | DsXorRtnB64 | DsWrxchgRtnB64 => (true, 1, true),
            DsMskorRtnB64 | DsCmpstRtnB64 | DsCmpstRtnF64 | DsMinRtnF64 | DsMaxRtnF64 | DsWrxchg2RtnB64
            | DsWrxchg2St64RtnB64 => (true, 2, true),
            _ => return None,
        })
    }

    /// Scalar loads (S_LOAD_*, S_BUFFER_LOAD_*): dwords loaded.
    pub fn scalar_load_dwords(self) -> u32 {
        use Opcode::*;
        match self {
            SLoadDword | SBufferLoadDword => 1,
            SLoadDwordx2 | SBufferLoadDwordx2 => 2,
            SLoadDwordx4 | SBufferLoadDwordx4 => 4,
            SLoadDwordx8 | SBufferLoadDwordx8 => 8,
            SLoadDwordx16 | SBufferLoadDwordx16 => 16,
            _ => 0,
        }
    }

    /// SMRD scalar scratch stores: dwords stored.
    pub fn scalar_store_dwords(self) -> u32 {
        match self {
            Opcode::ScratchStoreDword => 1,
            Opcode::ScratchStoreDwordx2 => 2,
            Opcode::ScratchStoreDwordx4 => 4,
            _ => 0,
        }
    }

    /// Dwords per lane moved by a MUBUF/MTBUF opcode (0 when unknown).
    pub fn buffer_dwords(self) -> u32 {
        use Opcode::*;
        match self {
            BufferLoadFormatX | BufferStoreFormatX => 1,
            BufferLoadFormatXy | BufferStoreFormatXy => 2,
            BufferLoadFormatXyz | BufferStoreFormatXyz => 3,
            BufferLoadFormatXyzw | BufferStoreFormatXyzw => 4,
            BufferLoadUbyte | BufferLoadSbyte | BufferLoadUshort | BufferLoadSshort | BufferStoreByte
            | BufferStoreByteD16Hi | BufferStoreShort | BufferStoreShortD16Hi | BufferLoadUbyteD16
            | BufferLoadUbyteD16Hi | BufferLoadSbyteD16 | BufferLoadSbyteD16Hi | BufferLoadShortD16
            | BufferLoadShortD16Hi => 1,
            BufferLoadDword | BufferStoreDword => 1,
            BufferLoadDwordx2 | BufferStoreDwordx2 => 2,
            BufferLoadDwordx3 | BufferStoreDwordx3 => 3,
            BufferLoadDwordx4 | BufferStoreDwordx4 => 4,
            BufferAtomicCmpswap | BufferAtomicSwapX2 | BufferAtomicOrX2 | BufferAtomicFcmpswap | BufferAtomicSubX2
            | BufferAtomicAddX2 | BufferAtomicSminX2 | BufferAtomicUminX2 | BufferAtomicSmaxX2 | BufferAtomicUmaxX2
            | BufferAtomicAndX2 | BufferAtomicXorX2 | BufferAtomicIncX2 | BufferAtomicDecX2 | BufferAtomicFminX2
            | BufferAtomicFmaxX2 => 2,
            BufferAtomicCmpswapX2 | BufferAtomicFcmpswapX2 => 4,
            BufferAtomicCsub => 1,
            BufferAtomicSwap | BufferAtomicAdd | BufferAtomicSub | BufferAtomicSmin | BufferAtomicUmin
            | BufferAtomicSmax | BufferAtomicUmax | BufferAtomicAnd | BufferAtomicOr | BufferAtomicXor
            | BufferAtomicInc | BufferAtomicDec | BufferAtomicFmin | BufferAtomicFmax => 1,
            TBufferLoadFormatX | TBufferStoreFormatX => 1,
            TBufferLoadFormatXy | TBufferStoreFormatXy => 2,
            TBufferLoadFormatXyz | TBufferStoreFormatXyz => 3,
            TBufferLoadFormatXyzw | TBufferStoreFormatXyzw => 4,
            TBufferLoadFormatD16X | TBufferLoadFormatD16Xy | TBufferStoreFormatD16X | TBufferStoreFormatD16Xy => 1,
            BufferLoadFormatD16X | BufferLoadFormatD16Xy | BufferStoreFormatD16X | BufferStoreFormatD16Xy
            | BufferLoadFormatD16HiX | BufferStoreFormatD16HiX => 1,
            BufferLoadFormatD16Xyz | BufferLoadFormatD16Xyzw | BufferStoreFormatD16Xyz | BufferStoreFormatD16Xyzw => 2,
            TBufferLoadFormatD16Xyz | TBufferLoadFormatD16Xyzw | TBufferStoreFormatD16Xyz
            | TBufferStoreFormatD16Xyzw => 2,
            _ => 0,
        }
    }

    /// Coarse class used for statistics.
    pub fn class(self) -> &'static str {
        let n = self.name();
        if n.starts_with("Image") {
            "image"
        } else if n.starts_with("Global") {
            "global_memory"
        } else if n.starts_with("VInterp") {
            "interp"
        } else if self == Opcode::Exp {
            "export"
        } else if n.starts_with("SLoad") || n.starts_with("SBufferLoad") {
            "scalar_load"
        } else if n.starts_with('V') {
            "valu"
        } else if n.starts_with('S') {
            "salu"
        } else {
            "other"
        }
    }
}

impl fmt::Display for Opcode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.mnemonic())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mnemonics() {
        assert_eq!(Opcode::SMovB32.mnemonic(), "s_mov_b32");
        assert_eq!(Opcode::SAndn2SaveexecB64.mnemonic(), "s_andn2_saveexec_b64");
        assert_eq!(Opcode::TBufferLoadFormatXyzw.mnemonic(), "tbuffer_load_format_xyzw");
        assert_eq!(Opcode::DsWrite2St64B32.mnemonic(), "ds_write2_st64_b32");
        assert_eq!(Opcode::Flat(FlatSegment::Global, FlatOp::LoadDwordx4).mnemonic(), "global_load_dwordx4");
        assert_eq!(Opcode::ImageSampleCLzO.mnemonic(), "image_sample_c_lz_o");
        assert_eq!(Opcode::VCmpxNeqF32.mnemonic(), "v_cmpx_neq_f32");
        assert_eq!(Opcode::Vop3Raw(0x1A5).name(), "Vop3Raw1A5");
    }

    #[test]
    fn names_are_unique() {
        let mut names: Vec<_> = Opcode::NAMED.iter().map(|o| o.name()).collect();
        names.sort();
        let len = names.len();
        names.dedup();
        assert_eq!(len, names.len());
    }
}
