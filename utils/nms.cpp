#include"gpu_nms.hpp"
#include<vector>
#include<iostream>
#include<torch/script.h>

torch::Tensor nms_cuda(int *keep_out,int *num_out,torch::Tensor & c_det,torch::Tensor & class_scores,int objs,float nms_overlap_thresh = 0.5)
{
	auto sort_ind = class_scores.sort(0,true);
	auto sorted = std::get<1>(sort_ind).accessor<long,2>();	

	auto dets = torch::zeros({objs,5});
	//ready for nms	
	for(size_t j =0;j<objs;j++)
	{
	   dets[j] = c_det[sorted[j][0]];
	}

	//nms
	_nms(keep_out,num_out,dets.data<float>(),objs,5,nms_overlap_thresh,0);
	return std::get<1>(sort_ind); 

}

