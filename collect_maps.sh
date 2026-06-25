for i in $(seq 0 17); do
    python examples/precompute_radio_maps.py \
        --hm DATA/radio_data_2/radio_data_2/hm/hm_${i}.npy \
        --out DATA_MAPS/radio_maps_hm_${i}
done