# -*- coding: utf-8 -*-
import numpy as np

def load_force_fields(base_dir, pot_dict, potential_run):
    print(f"Loading forces for {pot_dict}, run {potential_run}")
    # fine
    Fx_fine = np.load(f"{base_dir}/force_fine/{pot_dict}/force_fine_x_0{potential_run}.npy")
    Fy_fine = np.load(f"{base_dir}/force_fine/{pot_dict}/force_fine_y_0{potential_run}.npy")
    Fz_fine = np.load(f"{base_dir}/force_fine/{pot_dict}/force_fine_z_0{potential_run}.npy")

    # coarse
    Fx_coarse = np.load(f"{base_dir}/force_coarse/{pot_dict}/force_course_x_0{potential_run}.npy")
    Fy_coarse = np.load(f"{base_dir}/force_coarse/{pot_dict}/force_course_y_0{potential_run}.npy")
    Fz_coarse = np.load(f"{base_dir}/force_coarse/{pot_dict}/force_course_z_0{potential_run}.npy")

    # contiguous float32 (your exact intent)
    Fx_fine   = np.ascontiguousarray(Fx_fine.astype(np.float32, copy=False))
    Fy_fine   = np.ascontiguousarray(Fy_fine.astype(np.float32, copy=False))
    Fz_fine   = np.ascontiguousarray(Fz_fine.astype(np.float32, copy=False))
    Fx_coarse = np.ascontiguousarray(Fx_coarse.astype(np.float32, copy=False))
    Fy_coarse = np.ascontiguousarray(Fy_coarse.astype(np.float32, copy=False))
    Fz_coarse = np.ascontiguousarray(Fz_coarse.astype(np.float32, copy=False))
    Fx_coarse = np.ascontiguousarray(Fx_coarse.astype(np.float32, copy=False))
    Fy_coarse = np.ascontiguousarray(Fy_coarse.astype(np.float32, copy=False))
    Fz_coarse = np.ascontiguousarray(Fz_coarse.astype(np.float32, copy=False))

    return Fx_fine, Fy_fine, Fz_fine, Fx_coarse, Fy_coarse, Fz_coarse

'data/force_fine/potential_08/force_fine_x_08.npy'