cd build
cmake ..
echo "build app"
make -j4

cd ..
cd bin
echo "run app"
./RLR -d
