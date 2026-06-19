/// Natural column-major mapping (1-indexed): L(r, c; rows, cols) = r + (c - 1) * rows.
#[inline]
pub fn l_map(r: usize, c: usize, rows: usize, _cols: usize) -> usize {
    r + (c - 1) * rows
}

/// Natural column-major inverse mapping (1-indexed): L^-1(idx; rows, cols) = (row, col).
#[inline]
#[allow(dead_code)]
pub fn l_map_inv(idx: usize, rows: usize, _cols: usize) -> (usize, usize) {
    let r = (idx - 1) % rows + 1;
    let c = (idx - 1) / rows + 1;
    (r, c)
}

/// Inverse row-major mapping (1-indexed): L star map (r, c; rows, cols) = (r - 1) * cols + c.
#[inline]
#[allow(dead_code)]
pub fn l_star_map(r: usize, c: usize, _rows: usize, cols: usize) -> usize {
    (r - 1) * cols + c
}

/// Inverse row-major inverse mapping (1-indexed): (L star map)^-1(idx; rows, cols) = (row, col).
#[inline]
pub fn l_star_map_inv(idx: usize, _rows: usize, cols: usize) -> (usize, usize) {
    let r = (idx - 1) / cols + 1;
    let c = (idx - 1) % cols + 1;
    (r, c)
}
