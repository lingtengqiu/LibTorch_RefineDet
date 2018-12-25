/**********************************************************
 * Author        : lingteng qiu
 * Email         : 1259738366@qq.com
 * Create time   : 2018-12-25 14:36
 * Last modified : 2018-12-25 14:36
 * Filename      : post_process.cpp
 * Description   : post_process for our detection system 
 * *******************************************************/
#include<torch/script.h>
#include<iostream>
#include<vector>
#include<queue>
#include"post_process.hpp"

torch::Tensor jaccard(torch::Tensor source,torch::Tensor point)
{
	//step1 intersect

	torch::Tensor temp = torch::zeros({source.size(0),source.size(1)});
	
	temp.copy_(source);

	auto c1 = temp.select(1,0);
	auto c2 = temp.select(1,1);
	auto c3 = temp.select(1,2);
	auto c4 = temp.select(1,3);

	//area
	auto area_a = (c3-c1)*(c4-c2);

	auto area_b = (point[2]-point[0])*(point[3]-point[1]);

	//inter
	
	c1.masked_fill_(c1<point[0],point[0]);
	c2.masked_fill_(c2<point[1],point[1]);
	
	c3.masked_fill_(c3>point[2],point[2]);
	c4.masked_fill_(c4>point[3],point[3]);
	
	auto w = c3-c1;
	auto h = c4-c2;


	w.masked_fill_(w<0,0);
	h.masked_fill_(h<0,0);

	auto inter = w.mul(h);

	
	auto unions = area_a+ area_b-inter;


	return inter.div(unions);
}
torch::Tensor post_process(torch::Tensor boxes,std::tuple<int,int> shape)
{
	int h = std::get<0>(shape);		
	int w = std::get<1>(shape);
	auto temp = boxes.index_select(1,torch::tensor({0,1,2,3}).to(torch::kLong));
	temp.select(1,0).masked_fill_(temp.select(1,0)<0,0);
	temp.select(1,1).masked_fill_(temp.select(1,1)<0,0);
	temp.select(1,2).masked_fill_(temp.select(1,2)>=w,w-1);
	temp.select(1,3).masked_fill_(temp.select(1,3)>=h,h-1);
	std::vector<torch::Tensor> sets;
	while(temp.size(0)>0)
	{
		std::queue<torch::Tensor> q;
		q.push(temp.select(0,0));

		auto sub_set = q.front();

		temp = temp.slice(0,1,temp.size(0));
		while(!q.empty())
		{
			auto point = q.front();
			q.pop();
			auto overlap = jaccard(temp,point);
			std::vector<long> delete_list;
			for(int i = 0;i<overlap.size(0);i++)
			{
				if(overlap[i].item<float>() >0)
				{
					auto ele = temp.select(0,i);
					sub_set = torch::cat({sub_set,ele},0);
					q.push(ele);
				}
				else
				{
					delete_list.push_back(i);
				}
			}
			auto keep = torch::from_blob(delete_list.data(),{delete_list.size()},torch::kLong);
			temp = temp.index_select(0,keep);
		}
		sets.push_back(sub_set);	
	}	
	
	//find best over rect
	auto retv = torch::zeros({sets.size(),4});
	int cnt = 0;
	for(auto set : sets)
	{
		set =  set.view({-1,4});
		auto min_tensor = std::get<0>(set.min(0));
		auto max_tensor = std::get<0>(set.max(0));
		retv[cnt][0] = min_tensor[0];
		retv[cnt][1] = min_tensor[1];
		retv[cnt][2] = max_tensor[2];
		retv[cnt][3] = max_tensor[3];
		cnt++;
	}
	return retv;		


}
