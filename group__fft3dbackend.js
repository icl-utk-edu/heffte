var group__fft3dbackend =
[
    [ "cpu", "structheffte_1_1tag_1_1cpu.html", null ],
    [ "gpu", "structheffte_1_1tag_1_1gpu.html", null ],
    [ "data_manipulator", "structheffte_1_1backend_1_1data__manipulator.html", null ],
    [ "data_manipulator< tag::cpu >", "structheffte_1_1backend_1_1data__manipulator_3_01tag_1_1cpu_01_4.html", [
      [ "stream_type", "structheffte_1_1backend_1_1data__manipulator_3_01tag_1_1cpu_01_4.html#ac50f0e09631566154a1e93f8365f7836", null ]
    ] ],
    [ "is_enabled", "structheffte_1_1backend_1_1is__enabled.html", null ],
    [ "buffer_traits", "structheffte_1_1backend_1_1buffer__traits.html", [
      [ "location", "structheffte_1_1backend_1_1buffer__traits.html#a086cd1f9f7b4d1c4aacbf4c312a9d839", null ],
      [ "container", "structheffte_1_1backend_1_1buffer__traits.html#ada8b5c6138773c09facb96e7d099b011", null ]
    ] ],
    [ "uses_gpu", "structheffte_1_1backend_1_1uses__gpu.html", null ],
    [ "uses_gpu< backend_tag, typename std::enable_if< std::is_same< typename buffer_traits< backend_tag >::location, tag::gpu >::value, void >::type >", "structheffte_1_1backend_1_1uses__gpu_3_01backend__tag_00_01typename_01std_1_1enable__if_3_01std_6bad166a1b2e2b72cd8ee197f3565dff.html", null ],
    [ "device_instance", "structheffte_1_1backend_1_1device__instance.html", [
      [ "stream_type", "structheffte_1_1backend_1_1device__instance.html#a2b63f0b1deb23636cee0bfc94ffb564c", null ],
      [ "device_instance", "structheffte_1_1backend_1_1device__instance.html#a641aa853208673ca6458c1f9cfecc01c", null ],
      [ "~device_instance", "structheffte_1_1backend_1_1device__instance.html#ad405fb170c2b8d20c8d673ed414c4327", null ],
      [ "stream", "structheffte_1_1backend_1_1device__instance.html#a50b2c89b7100872ab6c704d3e4fb375e", null ],
      [ "stream", "structheffte_1_1backend_1_1device__instance.html#ad9c921523208ca3187ea2c755985d56c", null ],
      [ "synchronize_device", "structheffte_1_1backend_1_1device__instance.html#a1ffcdf4f88f7c5843296bb395a85a901", null ]
    ] ],
    [ "default_backend", "structheffte_1_1backend_1_1default__backend.html", [
      [ "type", "structheffte_1_1backend_1_1default__backend.html#a368d84d35a7d1d26d646b91b3dcb7ea2", null ]
    ] ],
    [ "uses_fft_types", "structheffte_1_1backend_1_1uses__fft__types.html", null ],
    [ "check_types", "structheffte_1_1backend_1_1check__types.html", null ],
    [ "check_types< backend_tag, input, output, typename std::enable_if< uses_fft_types< backend_tag >::value and((std::is_same< input, float >::value and is_ccomplex< output >::value) or(std::is_same< input, double >::value and is_zcomplex< output >::value) or(is_ccomplex< input >::value and is_ccomplex< output >::value) or(is_zcomplex< input >::value and is_zcomplex< output >::value))>::type >", "structheffte_1_1backend_1_1check__types_3_01backend__tag_00_01input_00_01output_00_01typename_01c412e8b081eebe81d371306e70d2da56.html", null ],
    [ "uses_fft_types< fftw_cos >", "structheffte_1_1backend_1_1uses__fft__types_3_01fftw__cos_01_4.html", null ],
    [ "uses_fft_types< fftw_sin >", "structheffte_1_1backend_1_1uses__fft__types_3_01fftw__sin_01_4.html", null ],
    [ "check_types< backend_tag, input, output, typename std::enable_if< not uses_fft_types< backend_tag >::value and((std::is_same< input, float >::value and std::is_same< output, float >::value) or(std::is_same< input, double >::value and std::is_same< output, double >::value))>::type >", "structheffte_1_1backend_1_1check__types_3_01backend__tag_00_01input_00_01output_00_01typename_01f9006ec2132705c3597d1dc98451f7b8.html", null ],
    [ "executor_base", "classheffte_1_1executor__base.html", [
      [ "~executor_base", "classheffte_1_1executor__base.html#a288af1a36ac40d341e18d614034978c7", null ],
      [ "forward", "classheffte_1_1executor__base.html#a53847ba79a69b44c28aa0af99ba886c4", null ],
      [ "forward", "classheffte_1_1executor__base.html#a6bed80e165a8d58f952b35f20b2b3c38", null ],
      [ "backward", "classheffte_1_1executor__base.html#a09fe4efecf5e28f891052399487c7c12", null ],
      [ "backward", "classheffte_1_1executor__base.html#ad3f50a316b629ebc5e7c8656cc4288a4", null ],
      [ "forward", "classheffte_1_1executor__base.html#a75c0b1773a9151e8855513c457febe44", null ],
      [ "forward", "classheffte_1_1executor__base.html#a15b90f516f94ab07c521ee84fff38db0", null ],
      [ "backward", "classheffte_1_1executor__base.html#ac6a079d83e77f115aa12bde838c173e2", null ],
      [ "backward", "classheffte_1_1executor__base.html#a6eb9b0d490fae5b0060763289a9142e9", null ],
      [ "forward", "classheffte_1_1executor__base.html#a25a520141f2ee711fc32d002eafebee2", null ],
      [ "forward", "classheffte_1_1executor__base.html#ad81c86e7a5411a146d077d305211eacb", null ],
      [ "backward", "classheffte_1_1executor__base.html#a98bf9ea775199c0055415042c70169fd", null ],
      [ "backward", "classheffte_1_1executor__base.html#a034923e5c12d5d01c677893600263d6e", null ],
      [ "box_size", "classheffte_1_1executor__base.html#ad61779bdc00afedbcc5799b9be70d1bc", null ],
      [ "workspace_size", "classheffte_1_1executor__base.html#a12de694aa4d15c4740e73dfb6b6cec74", null ],
      [ "complex_size", "classheffte_1_1executor__base.html#a09f83f4de9a869be73ac97cba917a312", null ]
    ] ],
    [ "one_dim_backend", "structheffte_1_1one__dim__backend.html", null ],
    [ "default_plan_options", "structheffte_1_1default__plan__options.html", null ],
    [ "name", "group__fft3dbackend.html#ga65d6597fd795ca487089c90a6052afab", null ],
    [ "name< tag::cpu >", "group__fft3dbackend.html#ga914051f89623d773cf98bd0373cc1e30", null ],
    [ "name< tag::gpu >", "group__fft3dbackend.html#ga5592e4064d57e5b1a15763dfd70b6200", null ],
    [ "make_buffer_container", "group__fft3dbackend.html#ga3d72a8e07b8a2a9638a962eda26e8ff6", null ],
    [ "has_executor2d", "group__fft3dbackend.html#gaf6f6fcf4321c67ab787419d0c5f4e307", null ],
    [ "has_executor3d", "group__fft3dbackend.html#gaf69364942863b80b82039ed78d7cf3d5", null ],
    [ "fft1d_get_howmany", "group__fft3dbackend.html#ga1fd8db18e1607f12b4e009abb8849626", null ],
    [ "fft1d_get_stride", "group__fft3dbackend.html#ga58cb5cd5676aeec4eecb94ff0111b440", null ]
];