for dir in $1/*
do
    if [[ -d $dir ]]; then
        echo "Processing: ${dir} ..."
        gin_config=$(find ${dir} -name "*.gin")
        echo "Using config: ${gin_config}..."
        sema_file="${dir}/_le_monstre_a_mange_ca_"
        if [[ -f "${sema_file}" ]]; then
            continue
        fi
        echo -n > ${sema_file}
        python main.py --gin_file=${gin_config} > ${dir}/c.out 2>&1
    fi
done