# pytorch_cpp_API
# LibTorch_RefineDet
first You shall compile the cuda_gpu for c++  
cd utils  
bash do.sh  
cp ./build/libgpu_nms.so ../  

second
compile your refinedet API  
bash do.sh  

# How To use
this model for 2 detector, if you want to add more.  
U only change a little from my code.  
You must train your refinedet in your pytorch,and import the torch script.  
and then U must give the anchor priors about U model and save to *.csv  

cd /my_master

U must put your test_img into JPEGImages.do follow:
U give a result dir for your test output
mkdir JPEGImages  
mkdir result

and get the filename save into OK.txt.  

# TEST
./build/launch [file_name.txt][U don't need][anchor.csv][Torch_model]  
