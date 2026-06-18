use fast_matmul::matmul::{l_map, l_map_inv};

#[test]
fn test_matlab_mappings() {
    // sub2ind([2, 3], 1, 1) = 1
    assert_eq!(l_map(1, 1, 2, 3), 1);
    // sub2ind([2, 3], 2, 1) = 2
    assert_eq!(l_map(2, 1, 2, 3), 2);
    // sub2ind([2, 3], 1, 2) = 3
    assert_eq!(l_map(1, 2, 2, 3), 3);
    // sub2ind([2, 3], 2, 3) = 6
    assert_eq!(l_map(2, 3, 2, 3), 6);

    // [r, c] = ind2sub([2, 3], 3) -> (1, 2)
    assert_eq!(l_map_inv(3, 2, 3), (1, 2));
    // [r, c] = ind2sub([2, 3], 6) -> (2, 3)
    assert_eq!(l_map_inv(6, 2, 3), (2, 3));
}
