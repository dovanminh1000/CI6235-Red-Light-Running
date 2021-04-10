cd build
rm -r *
echo "build RLR"
cmake ..
make -j4

echo "finish building"
