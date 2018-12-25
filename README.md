# pytorch_cpp_API
# LibTorch_RefineDet
first You shall compile the cuda_gpu for c++  

```BASH
cd utils  
bash do.sh  
cp ./build/libgpu_nms.so ../  
```


second compile your refinedet API
```BASH
cd ../
bash do.sh  
```
# How To use
this model for 2 detector, if you want to add more.  
U only change a little from my code.  
You must train your refinedet in your pytorch,and import the torch script.  
and then U must give the anchor priors about U model and save to *.csv  

## In your Master
U must put your test_img into JPEGImages.do follow:
U give a result dir for your test output
and get the filename save into file_name.txt. 
```BASH
mkdir JPEGImages  
mkdir result
``` 

# TEST
```BASH
./build/launch [file_name.txt][U don't need][anchor.csv][Torch_model]  
```

# FUTURE
`IF U have any question Email me.
