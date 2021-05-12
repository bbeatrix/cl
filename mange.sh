for dir in $2/*
do
    if [[ -d $dir ]]; then
        gin_config="${dir}/config.gin"
        sema_file="${dir}/_le_monstre_a_mange_ca_"
        if [[ -f "${sema_file}" ]]; then
            continue
        fi
        echo -n > ${sema_file}
        python $1 --gin_file=${gin_config} > ${dir}/cout 2> ${dir}/cerr
    fi
done