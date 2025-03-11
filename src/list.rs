use crate::builder::Builder;
use crate::interface::{ImmList, Interface, MutList};
use crate::interface_iter::{InterfaceIter, InterfaceIterCow};
use crate::iter::Iter;
use crate::level_iter::{LevelIter, LevelNode};
use crate::serde::ListVisitor;
use crate::tree::{IntraRebaseAction, RebaseAction};
use crate::update_map::MaxMap;
use crate::utils::{arb_arc, compute_level, int_log, opt_packing_depth, updated_length, Length};
use crate::{Arc, Cow, Error, Tree, UpdateMap, Value};
use arbitrary::Arbitrary;
use educe::Educe;
use itertools::process_results;
use serde::{ser::SerializeSeq, Deserialize, Deserializer, Serialize, Serializer};
use ssz::{Decode, Encode, SszEncoder, TryFromIter, BYTES_PER_LENGTH_OFFSET};
use std::collections::{BTreeMap, HashMap};
use std::marker::PhantomData;
use tree_hash::{Hash256, PackedEncoding, TreeHash};
use typenum::Unsigned;
use vec_map::VecMap;

#[derive(Debug, Clone, Educe, Arbitrary)]
#[educe(PartialEq(bound(T: Value, N: Unsigned, U: UpdateMap<T> + PartialEq)))]
#[arbitrary(bound = "T: Arbitrary<'arbitrary> + Value")]
#[arbitrary(bound = "N: Unsigned, U: Arbitrary<'arbitrary> + UpdateMap<T> + PartialEq")]
pub struct List<T: Value, N: Unsigned, U: UpdateMap<T> = MaxMap<VecMap<T>>> {
    pub(crate) interface: Interface<T, ListInner<T, N>, U>,
}

#[derive(Debug, Clone, Educe, Arbitrary)]
#[educe(PartialEq(bound(T: Value, N: Unsigned)))]
#[arbitrary(bound = "T: Arbitrary<'arbitrary> + Value, N: Unsigned")]
pub struct ListInner<T: Value, N: Unsigned> {
    #[arbitrary(with = arb_arc)]
    pub(crate) tree: Arc<Tree<T>>,
    pub(crate) length: Length,
    pub(crate) depth: usize,
    pub(crate) packing_depth: usize,
    #[arbitrary(default)]
    _phantom: PhantomData<N>,
}

impl<T: Value, N: Unsigned, U: UpdateMap<T>> List<T, N, U> {
    pub fn new(vec: Vec<T>) -> Result<Self, Error> {
        Self::try_from_iter(vec)
    }

    pub(crate) fn from_parts(tree: Arc<Tree<T>>, depth: usize, length: Length) -> Self {
        let packing_depth = opt_packing_depth::<T>().unwrap_or(0);
        Self {
            interface: Interface::new(ListInner {
                tree,
                length,
                depth,
                packing_depth,
                _phantom: PhantomData,
            }),
        }
    }

    pub fn empty() -> Self {
        // If the leaves are packed then they reduce the depth
        let depth = Self::depth();
        let tree = Tree::empty(depth);
        Self::from_parts(tree, depth, Length(0))
    }

    pub fn repeat(elem: T, n: usize) -> Result<Self, Error> {
        crate::repeat::repeat_list(elem, n)
    }

    pub fn repeat_slow(elem: T, n: usize) -> Result<Self, Error> {
        Self::try_from_iter(std::iter::repeat(elem).take(n))
    }

    pub fn builder() -> Result<Builder<T>, Error> {
        Builder::new(Self::depth(), 0)
    }

    pub fn try_from_iter(iter: impl IntoIterator<Item = T>) -> Result<Self, Error> {
        let mut builder = Self::builder()?;

        for item in iter.into_iter() {
            builder.push(item)?;
        }

        let (tree, depth, length) = builder.finish()?;

        // Check the length to cover the case where the capacity implied by packing_depth is
        // greater than N. E.g. the builder might pack up to 32 u8s, even if N is < 32.
        if length.as_usize() > N::to_usize() {
            return Err(Error::BuilderFull);
        }

        Ok(Self::from_parts(tree, depth, length))
    }

    /// This method exists for testing purposes.
    #[doc(hidden)]
    pub fn try_from_iter_slow(iter: impl IntoIterator<Item = T>) -> Result<Self, Error> {
        let mut list = Self::empty();

        for item in iter.into_iter() {
            list.push(item)?;
        }

        list.apply_updates()?;

        Ok(list)
    }

    pub fn to_vec(&self) -> Vec<T> {
        self.iter().cloned().collect()
    }

    pub fn iter(&self) -> InterfaceIter<T, U> {
        self.interface.iter()
    }

    pub fn iter_from(&self, index: usize) -> Result<InterfaceIter<T, U>, Error> {
        // Return an empty iterator at index == length, just like slicing.
        if index > self.len() {
            return Err(Error::OutOfBoundsIterFrom {
                index,
                len: self.len(),
            });
        }
        Ok(self.interface.iter_from(index))
    }

    /// Iterate all internal nodes on the same level as `index`.
    pub fn level_iter_from(&self, index: usize) -> Result<LevelIter<T>, Error> {
        // Return an empty iterator at index == length, just like slicing.
        if index > self.len() {
            return Err(Error::OutOfBoundsIterFrom {
                index,
                len: self.len(),
            });
        }
        self.interface.level_iter_from(index)
    }

    pub fn iter_cow(&mut self) -> InterfaceIterCow<T, U> {
        self.interface.iter_cow()
    }

    // Wrap trait methods so we present a Vec-like interface without having to import anything.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.interface.get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.interface.get_mut(index)
    }

    pub fn get_cow(&mut self, index: usize) -> Option<Cow<T>> {
        self.interface.get_cow(index)
    }

    pub fn push(&mut self, value: T) -> Result<(), Error> {
        self.interface.push(value)
    }

    pub fn len(&self) -> usize {
        self.interface.len()
    }

    pub fn is_empty(&self) -> bool {
        self.interface.is_empty()
    }

    pub fn has_pending_updates(&self) -> bool {
        self.interface.has_pending_updates()
    }

    pub fn apply_updates(&mut self) -> Result<(), Error> {
        self.interface.apply_updates()
    }

    pub fn bulk_update(&mut self, updates: U) -> Result<(), Error> {
        self.interface.bulk_update(updates)
    }

    pub(crate) fn depth() -> usize {
        if let Some(packing_bits) = opt_packing_depth::<T>() {
            int_log(N::to_usize()).saturating_sub(packing_bits)
        } else {
            int_log(N::to_usize())
        }
    }

    /// Remove `n` elements from the front of `self`.
    ///
    /// Errors if `n > self.len()`.
    pub fn pop_front_slow(&mut self, n: usize) -> Result<(), Error> {
        *self = Self::try_from_iter(self.iter_from(n)?.cloned())?;
        Ok(())
    }

    /// Remove `n` elements from the front of `self`.
    ///
    /// Errors if `n > self.len()`.
    pub fn pop_front(&mut self, n: usize) -> Result<(), Error> {
        self.apply_updates()?;

        if n == 0 {
            return Ok(());
        }

        let depth = Self::depth();
        let packing_depth = opt_packing_depth::<T>().unwrap_or(0);
        let level = compute_level(n, depth, packing_depth);
        let mut builder = Builder::new(Self::depth(), level)?;
        let mut level_iter = self.level_iter_from(n)?.peekable();

        while let Some(item) = level_iter.next() {
            match item {
                LevelNode::Internal(node) => {
                    let last = level_iter.peek().is_none();
                    let subtree_len = if !last {
                        1 << level
                    } else {
                        // Slower, but we only need to do this once.
                        node.compute_len()
                    };
                    builder.push_node(node.clone(), subtree_len)?;
                }
                LevelNode::PackedLeaf(value) => {
                    builder.push(value.clone())?;
                }
            }
        }

        let (tree, depth, length) = builder.finish()?;
        *self = Self::from_parts(tree, depth, length);

        Ok(())
    }
}

impl<T: Value, N: Unsigned> ImmList<T> for ListInner<T, N> {
    fn get(&self, index: usize) -> Option<&T> {
        if index < self.len().as_usize() {
            self.tree
                .get_recursive(index, self.depth, self.packing_depth)
        } else {
            None
        }
    }

    fn len(&self) -> Length {
        self.length
    }

    fn iter_from(&self, index: usize) -> Iter<T> {
        Iter::from_index(index, &self.tree, self.depth, self.length)
    }

    fn level_iter_from(&self, index: usize) -> LevelIter<T> {
        LevelIter::from_index(index, &self.tree, self.depth, self.length)
    }
}

impl<T, N> MutList<T> for ListInner<T, N>
where
    T: Value,
    N: Unsigned,
{
    fn validate_push(current_len: usize) -> Result<(), Error> {
        if current_len == N::to_usize() {
            Err(Error::ListFull { len: current_len })
        } else {
            Ok(())
        }
    }

    fn replace(&mut self, index: usize, value: T) -> Result<(), Error> {
        if index > self.len().as_usize() {
            return Err(Error::OutOfBoundsUpdate {
                index,
                len: self.len().as_usize(),
            });
        }

        self.tree = self.tree.with_updated_leaf(index, value, self.depth)?;
        if index == self.length.as_usize() {
            *self.length.as_mut() += 1;
        }
        Ok(())
    }

    fn update<U: UpdateMap<T>>(
        &mut self,
        updates: U,
        hash_updates: Option<BTreeMap<(usize, usize), Hash256>>,
    ) -> Result<(), Error> {
        if let Some(max_index) = updates.max_index() {
            if max_index >= N::to_usize() {
                return Err(Error::InvalidListUpdate);
            }
        } else {
            // Nothing to do.
            return Ok(());
        }
        self.length = updated_length(self.length, &updates);
        self.tree =
            self.tree
                .with_updated_leaves(&updates, 0, self.depth, hash_updates.as_ref())?;
        Ok(())
    }
}

impl<T: Value, N: Unsigned, U: UpdateMap<T>> List<T, N, U> {
    pub fn rebase(&self, base: &Self) -> Result<Self, Error> {
        let mut rebased = self.clone();
        rebased.rebase_on(base)?;
        Ok(rebased)
    }

    pub fn rebase_on(&mut self, base: &Self) -> Result<(), Error> {
        match Tree::rebase_on(
            &self.interface.backing.tree,
            &base.interface.backing.tree,
            Some((self.interface.backing.length, base.interface.backing.length)),
            self.interface.backing.depth + self.interface.backing.packing_depth,
        )? {
            RebaseAction::EqualReplace(replacement) => {
                self.interface.backing.tree = replacement.clone();
            }
            RebaseAction::NotEqualReplace(replacement) => {
                self.interface.backing.tree = replacement;
            }
            _ => (),
        }
        Ok(())
    }
}

impl<T: Value + Send + Sync, N: Unsigned, U: UpdateMap<T>> List<T, N, U> {
    pub fn intra_rebase(&mut self) -> Result<(), Error> {
        // We need to be fully hashed in order to intra-rebase. To avoid putting this burden on the
        // caller, just do it here. If we're already fully-hashed this should be quick.
        self.apply_updates()?;
        self.tree_hash_root();

        let mut known_subtrees = HashMap::new();
        if let IntraRebaseAction::Replace(new_tree) = Tree::intra_rebase(
            &self.interface.backing.tree,
            &mut known_subtrees,
            self.interface.backing.depth,
        )? {
            self.interface.backing.tree = new_tree;
        }
        Ok(())
    }

    /// Prunes the tree by replacing unnecessary branches with their hashes.
    /// The tree structure and depth remain the same, but branches that don't lead
    /// to indices in the input are replaced with their hash values.
    ///
    /// Note: The tree must be fully hashed before calling this function.
    /// You should call `apply_updates()` and ensure the tree is hashed before pruning.
    ///
    /// # Arguments
    /// * `indices` - Vector of indices to keep fully expanded in the tree
    ///
    /// # Returns
    /// * `Result<(), Error>` - Ok if pruning successful, Error if any index is invalid
    pub fn prune(&mut self, indices: Vec<usize>, default_value: T) -> Result<(), Error> {
        // Validate indices
        if let Some(&max_index) = indices.iter().max() {
            if max_index >= self.len() {
                return Err(Error::OutOfBoundsUpdate {
                    index: max_index,
                    len: self.len(),
                });
            }
        }

        // Sort and deduplicate indices
        let mut unique_indices: Vec<_> = indices.into_iter().collect();
        unique_indices.sort_unstable();
        unique_indices.dedup();

        // Create a set of indices for faster lookup
        let indices_set: std::collections::HashSet<_> = unique_indices.into_iter().collect();

        // Prune the tree while maintaining structure
        let pruned_tree = self.interface.backing.tree.prune_keeping_structure(
            &indices_set,
            self.interface.backing.depth,
            0,
            default_value,
            true,
        )?;

        // Update the tree while keeping other properties the same
        self.interface.backing.tree = pruned_tree;

        Ok(())
    }
}

impl<T: Value, N: Unsigned, U: UpdateMap<T>> Default for List<T, N, U> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<T: Value + Send + Sync, N: Unsigned, U: UpdateMap<T>> TreeHash for List<T, N, U> {
    fn tree_hash_type() -> tree_hash::TreeHashType {
        tree_hash::TreeHashType::List
    }

    fn tree_hash_packed_encoding(&self) -> PackedEncoding {
        unreachable!("List should never be packed.")
    }

    fn tree_hash_packing_factor() -> usize {
        unreachable!("List should never be packed.")
    }

    fn tree_hash_root(&self) -> Hash256 {
        // FIXME(sproul): remove assert
        assert!(!self.interface.has_pending_updates());

        let root = self.interface.backing.tree.tree_hash();
        tree_hash::mix_in_length(&root, self.len())
    }
}

impl<'a, T: Value, N: Unsigned, U: UpdateMap<T>> IntoIterator for &'a List<T, N, U> {
    type Item = &'a T;
    type IntoIter = InterfaceIter<'a, T, U>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T: Value, N: Unsigned, U: UpdateMap<T>> Serialize for List<T, N, U>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.len()))?;
        for e in self {
            seq.serialize_element(e)?;
        }
        seq.end()
    }
}

impl<'de, T, N, U> Deserialize<'de> for List<T, N, U>
where
    T: Deserialize<'de> + Value,
    N: Unsigned,
    U: UpdateMap<T>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(ListVisitor::default())
    }
}

// FIXME: duplicated from `ssz::encode::impl_for_vec`
impl<T: Value, N: Unsigned, U: UpdateMap<T>> Encode for List<T, N, U> {
    fn is_ssz_fixed_len() -> bool {
        false
    }

    fn ssz_bytes_len(&self) -> usize {
        if <T as Encode>::is_ssz_fixed_len() {
            <T as Encode>::ssz_fixed_len() * self.len()
        } else {
            let mut len = self.iter().map(|item| item.ssz_bytes_len()).sum();
            len += BYTES_PER_LENGTH_OFFSET * self.len();
            len
        }
    }

    fn ssz_append(&self, buf: &mut Vec<u8>) {
        if <T as Encode>::is_ssz_fixed_len() {
            buf.reserve(<T as Encode>::ssz_fixed_len() * self.len());

            for item in self {
                item.ssz_append(buf);
            }
        } else {
            let mut encoder = SszEncoder::container(buf, self.len() * BYTES_PER_LENGTH_OFFSET);

            for item in self {
                encoder.append(item);
            }

            encoder.finalize();
        }
    }
}

impl<T, N, U> TryFromIter<T> for List<T, N, U>
where
    T: Value,
    N: Unsigned,
    U: UpdateMap<T>,
{
    type Error = Error;

    fn try_from_iter<I>(iter: I) -> Result<Self, Self::Error>
    where
        I: IntoIterator<Item = T>,
    {
        List::try_from_iter(iter)
    }
}

impl<T, N, U> Decode for List<T, N, U>
where
    T: Value,
    N: Unsigned,
    U: UpdateMap<T>,
{
    fn is_ssz_fixed_len() -> bool {
        false
    }

    fn from_ssz_bytes(bytes: &[u8]) -> Result<Self, ssz::DecodeError> {
        let max_len = N::to_usize();

        if bytes.is_empty() {
            Ok(List::empty())
        } else if <T as Decode>::is_ssz_fixed_len() {
            let num_items = bytes
                .len()
                .checked_div(<T as Decode>::ssz_fixed_len())
                .ok_or(ssz::DecodeError::ZeroLengthItem)?;

            if num_items > max_len {
                return Err(ssz::DecodeError::BytesInvalid(format!(
                    "List of {} items exceeds maximum of {}",
                    num_items, max_len
                )));
            }

            process_results(
                bytes
                    .chunks(<T as Decode>::ssz_fixed_len())
                    .map(T::from_ssz_bytes),
                |iter| {
                    List::try_from_iter(iter).map_err(|e| {
                        ssz::DecodeError::BytesInvalid(format!("Error building ssz List: {:?}", e))
                    })
                },
            )?
        } else {
            ssz::decode_list_of_variable_length_items(bytes, Some(max_len))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::update_map::MaxMap;
    use alloy_primitives::Uint;
    use typenum::U8388608;
    use vec_map::VecMap;

    #[test]
    fn test_prune() {
        // Create a list with numbers 1 to 3213
        let numbers: Vec<Uint<256, 4>> = (0..3213).map(|i| Uint::<256, 4>::from(i + 1)).collect();

        // Create two identical lists - one to prune, one to keep full
        let mut pruned_list: List<Uint<256, 4>, U8388608, MaxMap<VecMap<Uint<256, 4>>>> =
            List::new(numbers.clone()).unwrap();
        let mut full_list: List<Uint<256, 4>, U8388608, MaxMap<VecMap<Uint<256, 4>>>> =
            List::new(numbers).unwrap();

        // Ensure both trees are fully hashed before pruning
        pruned_list.apply_updates().unwrap();
        full_list.apply_updates().unwrap();

        // Get initial hash
        let initial_hash = pruned_list.tree_hash_root();
        assert_eq!(
            full_list.tree_hash_root(),
            initial_hash,
            "Initial hashes should match"
        );

        println!("Original tree structure:");
        println!("tree hash root: {:?}", initial_hash);
        println!("tree depth: {:?}", pruned_list.interface.backing.depth);
        println!("{:#?}", pruned_list.interface.backing.tree);

        // Keep only indices 123, 989, and 3065 in the pruned list
        pruned_list
            .prune(vec![123, 989, 3065], Uint::<256, 4>::from(0))
            .unwrap();

        println!("\nPruned tree structure (keeping indices 123, 989, 3065):");
        println!("{:#?}", pruned_list.interface.backing.tree);

        // Verify the tree structure is maintained
        assert_eq!(pruned_list.len(), 3213, "Length should remain the same");
        assert_eq!(
            pruned_list.interface.backing.depth, full_list.interface.backing.depth,
            "Depth should remain the same"
        );

        // Verify we can still access the kept indices
        assert_eq!(pruned_list.get(123).unwrap(), &Uint::<256, 4>::from(124));
        assert_eq!(pruned_list.get(989).unwrap(), &Uint::<256, 4>::from(990));
        assert_eq!(pruned_list.get(3065).unwrap(), &Uint::<256, 4>::from(3066));

        // Update value at index 989 in both lists
        let new_value = Uint::<256, 4>::from(42);
        pruned_list
            .interface
            .backing
            .replace(989, new_value.clone())
            .unwrap();
        full_list.interface.backing.replace(989, new_value).unwrap();

        // Apply updates to ensure hashes are recalculated
        pruned_list.apply_updates().unwrap();
        full_list.apply_updates().unwrap();

        // Get new hashes
        let pruned_hash = pruned_list.tree_hash_root();
        let full_hash = full_list.tree_hash_root();

        println!("\nAfter update at index 989:");
        println!("Pruned tree hash: {:?}", pruned_hash);
        println!("Full tree hash: {:?}", full_hash);

        println!("Pruned tree len: {}", pruned_list.interface.backing.tree.compute_len());
        println!("Full tree len: {}", full_list.interface.backing.tree.compute_len());

        // Verify hashes match after update
        assert_eq!(pruned_hash, full_hash, "Hashes should match after update");
        assert_ne!(pruned_hash, initial_hash, "Hash should change after update");

        // Verify the updated value is accessible in both lists
        assert_eq!(pruned_list.get(989).unwrap(), &Uint::<256, 4>::from(42));
        assert_eq!(full_list.get(989).unwrap(), &Uint::<256, 4>::from(42));
    }
}
