use ndarray::ArrayView1;

#[no_mangle]
pub extern "C" fn print_array(ptr: *const f64, len: usize) {
    let array = unsafe {
        assert!(!ptr.is_null());
        ArrayView1::from_shape_ptr(len, ptr)
    };

    for &item in array.iter() {
        println!("{}", item);
    }
}
