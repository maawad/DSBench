num_keys=50000000
#num_keys=1000000

device=0

load_factor=(0.60 0.65 0.70 0.75 0.80 0.82 0.84 0.86 0.88 0.90 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99)

for lf in "${load_factor[@]}"
do
    echo ./build/bin/test_cuco --device=$device --num-keys=$num_keys --load-factor=$lf
    ./build/bin/test_cuco --device=$device --num-keys=$num_keys --load-factor=$lf
done