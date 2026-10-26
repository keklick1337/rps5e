// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Sparse two-level page table: an entry per guest page, selected by page
//! number. Second-level buckets are allocated on the first write into them;
//! lookups never allocate. O(1) owner lookups for the caches (the buffer of
//! a caching page, the images over a 1 MiB page).
//!
//! [`InlinePageOwnerList`] keeps the owners of one page: the first few
//! inline, more on the heap.

/// `PAGE_BITS`-sized pages over an `ADDRESS_BITS` address space, with
/// `FIRST_BITS` bits of page number selecting the bucket.
pub struct MultiLevelPageTable<E: Default, const PAGE_BITS: u32, const ADDRESS_BITS: u32, const FIRST_BITS: u32> {
    first: Vec<Option<Box<[E]>>>,
    allocated: usize,
}

impl<E: Default, const PAGE_BITS: u32, const ADDRESS_BITS: u32, const FIRST_BITS: u32> Default
    for MultiLevelPageTable<E, PAGE_BITS, ADDRESS_BITS, FIRST_BITS>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<E: Default, const PAGE_BITS: u32, const ADDRESS_BITS: u32, const FIRST_BITS: u32> MultiLevelPageTable<E, PAGE_BITS, ADDRESS_BITS, FIRST_BITS> {
    const SECOND_BITS: u32 = ADDRESS_BITS - FIRST_BITS - PAGE_BITS;
    pub const BUCKET_ENTRIES: usize = 1 << Self::SECOND_BITS;
    pub const PAGE_COUNT: u64 = 1 << (ADDRESS_BITS - PAGE_BITS);
    pub const PAGE_BITS: u32 = PAGE_BITS;
    pub const ADDRESS_SPACE: u64 = 1 << ADDRESS_BITS;

    pub fn new() -> Self {
        const { assert!(ADDRESS_BITS < 64 && PAGE_BITS < ADDRESS_BITS && FIRST_BITS > 0 && FIRST_BITS <= ADDRESS_BITS - PAGE_BITS) };
        Self { first: (0..1usize << FIRST_BITS).map(|_| None).collect(), allocated: 0 }
    }

    pub fn is_valid_page(page: u64) -> bool {
        page < Self::PAGE_COUNT
    }

    /// The entry of `page` (None: outside the space or never written).
    #[inline]
    pub fn find(&self, page: u64) -> Option<&E> {
        if !Self::is_valid_page(page) {
            return None;
        }
        self.first[(page >> Self::SECOND_BITS) as usize].as_ref().map(|b| &b[page as usize & (Self::BUCKET_ENTRIES - 1)])
    }

    /// The entry of `page` for changes, without allocating (None: outside
    /// the space or never written).
    #[inline]
    pub fn find_mut(&mut self, page: u64) -> Option<&mut E> {
        if !Self::is_valid_page(page) {
            return None;
        }
        self.first[(page >> Self::SECOND_BITS) as usize].as_mut().map(|b| &mut b[page as usize & (Self::BUCKET_ENTRIES - 1)])
    }

    /// The entry of `page`, its bucket allocated when missing (None:
    /// outside the space).
    pub fn get_or_create(&mut self, page: u64) -> Option<&mut E> {
        if !Self::is_valid_page(page) {
            return None;
        }
        let bucket = &mut self.first[(page >> Self::SECOND_BITS) as usize];
        if bucket.is_none() {
            *bucket = Some((0..Self::BUCKET_ENTRIES).map(|_| E::default()).collect());
            self.allocated += 1;
        }
        bucket.as_mut().map(|b| &mut b[page as usize & (Self::BUCKET_ENTRIES - 1)])
    }

    /// The half-open page interval a non-empty byte range touches (None:
    /// empty, wrapping or outside the space).
    pub fn page_range(address: u64, size: u64) -> Option<std::ops::Range<u64>> {
        if size == 0 || address >= Self::ADDRESS_SPACE || size > Self::ADDRESS_SPACE - address {
            return None;
        }
        Some(address >> PAGE_BITS..((address + size - 1) >> PAGE_BITS) + 1)
    }

    pub fn allocated_buckets(&self) -> usize {
        self.allocated
    }
}

/// The owners of one page, in registration order: up to `N` inline, more
/// on the heap (and back inline once they fit again).
#[derive(Debug)]
pub struct InlinePageOwnerList<T: Copy + Default + PartialEq, const N: usize> {
    inline: [T; N],
    len: usize,
    overflow: Option<Box<Vec<T>>>,
}

impl<T: Copy + Default + PartialEq, const N: usize> Default for InlinePageOwnerList<T, N> {
    fn default() -> Self {
        InlinePageOwnerList { inline: [T::default(); N], len: 0, overflow: None }
    }
}

impl<T: Copy + Default + PartialEq, const N: usize> InlinePageOwnerList<T, N> {
    pub fn as_slice(&self) -> &[T] {
        match &self.overflow {
            Some(v) => v,
            None => &self.inline[..self.len],
        }
    }

    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn front(&self) -> Option<T> {
        self.as_slice().first().copied()
    }

    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        self.as_slice().iter().copied()
    }

    pub fn contains(&self, owner: T) -> bool {
        self.as_slice().contains(&owner)
    }

    pub fn push(&mut self, owner: T) {
        if let Some(v) = &mut self.overflow {
            v.push(owner);
        } else if self.len < N {
            self.inline[self.len] = owner;
            self.len += 1;
        } else {
            let mut v = Vec::with_capacity(N * 2);
            v.extend_from_slice(&self.inline);
            v.push(owner);
            self.overflow = Some(Box::new(v));
        }
    }

    /// Remove `owner`; false when it is not registered.
    pub fn erase(&mut self, owner: T) -> bool {
        if let Some(v) = &mut self.overflow {
            let Some(i) = v.iter().position(|&o| o == owner) else { return false };
            v.remove(i);
            if v.len() == N {
                self.inline.copy_from_slice(v);
                self.len = N;
                self.overflow = None;
            }
            return true;
        }
        let Some(i) = self.inline[..self.len].iter().position(|&o| o == owner) else { return false };
        self.inline.copy_within(i + 1..self.len, i);
        self.len -= 1;
        true
    }

    /// Move the owners out, leaving the list empty.
    pub fn take(&mut self) -> Self {
        std::mem::take(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Table = MultiLevelPageTable<u64, 14, 40, 14>;

    #[test]
    fn lookups_do_not_allocate_and_writes_stick() {
        let mut t = Table::new();
        assert!(t.find(5).is_none());
        assert_eq!(t.allocated_buckets(), 0);
        *t.get_or_create(5).unwrap() = 7;
        assert_eq!(t.find(5), Some(&7));
        assert_eq!(t.find(6), Some(&0));
        assert_eq!(t.allocated_buckets(), 1);
        assert!(t.get_or_create(1 << 26).is_none());
    }

    #[test]
    fn page_ranges() {
        assert_eq!(Table::page_range(0x4000, 1), Some(1..2));
        assert_eq!(Table::page_range(0x3fff, 2), Some(0..2));
        assert_eq!(Table::page_range(0, 0), None);
        assert_eq!(Table::page_range(Table::ADDRESS_SPACE - 0x4000, 0x4000), Some((1 << 26) - 1..1 << 26));
        assert_eq!(Table::page_range(Table::ADDRESS_SPACE - 0x4000, 0x4001), None);
    }

    // The image page-table cases of the reference suite.
    type BigTable = MultiLevelPageTable<Vec<u32>, 20, 44, 14>;
    type PageOwners = InlinePageOwnerList<u32, 16>;
    type OwnerTable = MultiLevelPageTable<PageOwners, 20, 44, 14>;

    #[test]
    fn multi_owner_and_exact_erase() {
        let mut table = OwnerTable::new();
        let owners = table.get_or_create(17).unwrap();
        owners.push(11);
        owners.push(22);
        assert_eq!(table.find(17).unwrap().len(), 2, "both page owners are retained");
        let owners = table.find_mut(17).unwrap();
        assert!(owners.erase(11), "registered owner is erased");
        assert_eq!((owners.len(), owners.front()), (1, Some(22)), "erasing one owner preserves its neighbor");
        assert!(!owners.erase(33), "missing owner is reported without mutation");
    }

    #[test]
    fn cross_bucket_range() {
        let boundary = (BigTable::BUCKET_ENTRIES as u64) << BigTable::PAGE_BITS;
        let range = BigTable::page_range(boundary - 1, 2).unwrap();
        assert_eq!(range, BigTable::BUCKET_ENTRIES as u64 - 1..BigTable::BUCKET_ENTRIES as u64 + 1);
        let mut table = BigTable::new();
        table.get_or_create(range.start).unwrap().push(1);
        table.get_or_create(range.end - 1).unwrap().push(2);
        assert_eq!(table.allocated_buckets(), 2, "pages across the L1 boundary use distinct sparse buckets");
    }

    #[test]
    fn queries_do_not_allocate() {
        let mut table = BigTable::new();
        assert!(table.find(123).is_none());
        assert!(table.find_mut(123).is_none());
        assert!(table.find(BigTable::PAGE_COUNT - 1).is_none());
        assert!(table.find(BigTable::PAGE_COUNT).is_none());
        assert_eq!(table.allocated_buckets(), 0);
    }

    #[test]
    fn address_space_boundaries() {
        let space = BigTable::ADDRESS_SPACE;
        assert_eq!(BigTable::page_range(space - 1, 1), Some(BigTable::PAGE_COUNT - 1..BigTable::PAGE_COUNT), "last guest byte maps to the final page");
        assert_eq!(BigTable::page_range(0, 0), None, "empty ranges are rejected");
        assert_eq!(BigTable::page_range(space, 1), None, "first out-of-range byte is rejected");
        assert_eq!(BigTable::page_range(space - 1, 2), None, "crossing the address-space end is rejected");
        assert_eq!(BigTable::page_range(u64::MAX - 1, 4), None, "wrapping input is rejected");
        let mut table = BigTable::new();
        table.get_or_create(BigTable::PAGE_COUNT - 1).unwrap().push(99);
        assert_eq!(table.find(BigTable::PAGE_COUNT - 1).unwrap()[0], 99);
    }

    #[test]
    fn inline_owner_storage_and_overflow() {
        let mut owners = PageOwners::default();
        for o in 1..=18 {
            owners.push(o);
        }
        assert_eq!((owners.len(), owners.front()), (18, Some(1)));
        assert!(owners.iter().eq(1..=18), "overflow preserves registration order");
        assert!(owners.erase(5) && owners.len() == 17 && !owners.contains(5));
        assert!(owners.erase(18) && owners.len() == 16, "overflow storage shrinks back to inline capacity");
        assert_eq!(owners.as_slice(), &[1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]);
        assert!(!owners.erase(99));
        let moved = owners.take();
        assert!(owners.is_empty() && moved.len() == 16 && moved.front() == Some(1));
        let mut partial = PageOwners::default();
        partial.push(41);
        partial.push(42);
        let moved = partial.take();
        assert!(partial.is_empty() && moved.as_slice() == [41, 42]);
    }

    #[test]
    fn one_mib_registration_granularity() {
        assert_eq!(OwnerTable::PAGE_BITS, 20);
        let mut table = OwnerTable::new();
        let pages = OwnerTable::page_range(0, 64 << 20).unwrap();
        assert_eq!(pages, 0..64, "64 MiB registration touches exactly 64 one-MiB entries");
        for p in pages.clone() {
            table.get_or_create(p).unwrap().push(7);
        }
        assert_eq!(table.allocated_buckets(), 1);
        for p in pages {
            let owners = table.find_mut(p).unwrap();
            assert_eq!((owners.len(), owners.front()), (1, Some(7)));
            assert!(owners.erase(7));
        }
    }

    #[test]
    fn shared_coarse_page_lifecycle() {
        let mut table = OwnerTable::new();
        let owners = table.get_or_create(2).unwrap();
        owners.push(11);
        owners.push(22);
        assert!(owners.erase(11) && owners.as_slice() == [22]);
        assert!(!owners.erase(11), "double unregister is rejected without mutation");
        assert!(owners.erase(22) && owners.is_empty());
    }

    #[test]
    fn one_mib_boundaries() {
        assert_eq!(OwnerTable::page_range(0x0F_FFFF, 2), Some(0..2));
        assert_eq!(OwnerTable::page_range(OwnerTable::ADDRESS_SPACE - 1, 1), Some(OwnerTable::PAGE_COUNT - 1..OwnerTable::PAGE_COUNT));
    }
}
