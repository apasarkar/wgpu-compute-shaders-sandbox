@group(0) @binding(0)
var<storage, read> in_data: array<i32>;

//For now provide run length as an explicit sequence of visition values in dim1
@group(0) @binding(1)
var<storage, read> visitation_dim1: array<u32>;

//For now provide run length as an explicit sequence of visition values in dim2
@group(0) @binding(2)
var<storage, read> visitation_dim2: array<u32>;

@group(0) @binding(3)
var<storage,read_write> total_length: array<u32>;

const n_rows = u32(512);  // m_rows
const n_cols = u32(512);  // n_cols



@compute
@workgroup_size(1, 1, 1)
fn main(@builtin(workgroup_id) wg_id : vec3<u32>,
        @builtin(num_workgroups) n_wkgp: vec3<u32>) {
    var wkgp_d1: u32=n_wkgp.y;
    var wkgp_d2: u32=n_wkgp.x;
    var d1: u32=wg_id.y;
    var d2: u32=wg_id.x;

    var start_pt_d1: u32=d1*8;
    var start_pt_d2: u32=d2*8;

    var current_wkgp_index: u32 = (d1*wkgp_d2 + d2);
   // var generic_start_index: u32=  current_wkgp_index * 128;


    var is_zeros: bool = false;
    var counter: u32 = 0;
    var save_counter: u32 = 0;
    for (var i: u32 = 0; i < 64; i++) {
        // For now we run length encode everything, including the DC component
        // First get the integer value
        var dim1_displacement: u32 = visitation_dim1[i];
        var dim2_displacement: u32 = visitation_dim2[i];
        var curr_ind: u32 = (start_pt_d1 + dim1_displacement) * n_cols + (start_pt_d2 + dim2_displacement);
        var curr_value = in_data[curr_ind];

        if (curr_value == 0) {
            if (is_zeros) {
                counter += 1;
                
            } else {
                is_zeros = true;
                counter = 1;
            }
        } else {
            if (is_zeros) {
                is_zeros = false;
                while (counter > 16) {
                    // out_data[generic_start_index + save_counter] = 15;
                    save_counter += 1;
                    // out_data[generic_start_index + save_counter] = 0;
                    save_counter += 1;
                    counter -= 16;
                }
                if (counter > 0) {
                    // out_data[generic_start_index + save_counter] = counter;
                    save_counter += 1;
                    // out_data[generic_start_index + save_counter] = curr_value;
                    save_counter += 1;
                }
                counter = 0;
            } else {
                // out_data[generic_start_index + save_counter] = 0;
                save_counter += 1;
                // out_data[generic_start_index + save_counter] = curr_value;
                save_counter += 1;
                is_zeros = false;
                counter = 0;
            }

        }


    }
    // If we are writing out an explicit save counter, we don't need EOF delimiters
    // out_data[generic_start_index + save_counter] = 132435;
    // save_counter += 1; 
    // out_data[generic_start_index + save_counter] = 132435;
    total_length[current_wkgp_index] = save_counter;
}