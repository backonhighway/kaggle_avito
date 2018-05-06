
def get_dtypes():
    dtypes = {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8',
        'click_id': 'int32'
    }
    return dtypes


def get_featured_dtypes():

    dtypes = {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8',
        'click_id': 'int32',
        'hour': 'uint16',
        'app_count': 'uint32',
        'os_count': 'uint32',
        'idoa_is_last_try': 'uint8',
        'ioa_is_last_try': 'uint8',
        'io_is_last_try': 'uint8',
        "group_i_hourly_count": "uint32",
        "group_i_hourly_count_share": "float32",
        'group_i_count': 'uint32',
        'group_ia_count': 'uint32',
        'group_io_count': 'uint32',
        'group_ic_count': 'uint32',
        'group_ido_count': 'uint32',
        'group_ioa_count': 'uint32',
        'group_idoa_count': 'uint32',
        'group_iac_count': 'uint32',
        'group_ioc_count': 'uint32',
        'group_ioac_count': 'uint32',
        'group_idoac_count': 'uint32',
        'group_i_nunique_os': 'uint32',
        'group_i_nunique_app': 'uint32',
        'group_i_nunique_channel': 'uint32',
        'group_i_nunique_device': 'uint32',
        'group_i_prev_click_time': 'float32',
        'group_i_next_click_time': 'float32',
        'group_ict_max': 'float32',
        'group_ict_std': 'float32',
        'group_ict_mean': 'float32',
        'group_ict_sum': 'float32',
        'group_io_nunique_app': 'uint32',
        'group_io_nunique_channel': 'uint32',
        'group_io_prev_click_time': 'float32',
        'group_io_next_click_time': 'float32',
        'group_ioct_max': 'float32',
        'group_ioct_std': 'float32',
        'group_ioct_mean': 'float32',
        'group_ioct_sum': 'float32',
        'group_ido_nunique_app': 'uint32',
        'group_ido_nunique_channel': 'uint32',
        'group_ido_prev_click_time': 'float32',
        'group_ido_next_click_time': 'float32',
        'group_idoct_max': 'float32',
        'group_idoct_std': 'float32',
        'group_idoct_mean': 'float32',
        'group_idoct_sum': 'float32',
        "group_i_top1_device_share": 'int16',
        "group_i_top2_device_share": 'int16',
        "group_ido_rolling_mean_prev_ct": 'float32'
    }
    return dtypes