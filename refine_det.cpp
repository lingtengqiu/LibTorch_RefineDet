/**********************************************************
 * Author        : lingteng qiu
 * Email         : 
 * Create time   : 2018-12-25 22:17
 * Last modified : 2018-12-25 22:17
 * Filename      : resnet_class.cpp
 * Description   : a example about refineDet for two detector  
 * *******************************************************/
#include<torch/script.h>
#include<math.h>
#include<memory>
#include<cstddef>
#include<iostream>
#include<string>
#include<vector>
#include<torch/torch.h>
#include<assert.h>
#include<opencv2/opencv.hpp>
#include<unordered_map>
#include<string>
#include<sstream>
#include<istream>
#include<typeinfo>
#include"nms.hpp"
#include"post_process.hpp"
//#define TEST 1 
//this part is using the options about this system
#define POSFIX ".bmp"
struct Options{
	std::string data_root{"data"};
	bool no_cuda{false};
	int32_t seed{1};
	int32_t test_batch_size{1000};
	int32_t log_interval{10};
	int32_t num_class{2};
	float thres{0.5};
};

void forward_region_priors(std::string & dir,std::vector<float> &region_priors)
{
	FILE *f = std::fopen(dir.c_str(),"r");		
	string temp;
	
	float a,b,c,d;
	while(1)
	{
		std::fscanf(f,"%f,%f,%f,%f",&a,&b,&c,&d);
		region_priors.push_back(a);
		region_priors.push_back(b);
		region_priors.push_back(c);
		region_priors.push_back(d);
		if(std::feof(f))
			break;
	}		
	region_priors.erase(region_priors.begin(),region_priors.begin()+4);
	std::cout<<region_priors.size()<<std::endl;


}
struct detector
{
	int num_class;
	float obj_threshed;
	std::vector<float> variance;

	detector(int n,float thre=0.05):num_class(n),obj_threshed(thre),variance({0.1,0.2})
	{
			
	}
	std::tuple<torch::Tensor,torch::Tensor> operator()(torch::Tensor &odm_loc,torch::Tensor &odm_conf,torch::Tensor & prior,torch::Tensor &arm_loc,torch::Tensor & arm_conf)
	{
		long batch = odm_loc.size(0);
		//according to arm filter the data
		auto filter_odm_conf = torch::zeros({16320,2}).cuda();	
		filter_odm_conf= odm_conf.where(arm_conf>obj_threshed,filter_odm_conf);

		long num_priors = prior.size(0);
		auto boxes = torch::zeros({batch,num_priors,4}).cuda();

		auto scores = torch::zeros({batch,num_priors,num_class}).cuda();
		auto conf_preds = filter_odm_conf.view({batch,num_priors,num_class});
		torch::Tensor _default;
		torch::Tensor decode_boxes;

		for(int i = 0;i<batch;i++)
		{
			_default = decode(arm_loc[i],prior);

			_default = center(_default);

			decode_boxes = decode(odm_loc[i],_default);
			boxes[i] = decode_boxes;
			scores[i] = filter_odm_conf.clone();
		}
		std::tuple<torch::Tensor,torch::Tensor> retv(boxes.cpu(),scores.cpu());
		return retv;
		
	}
	torch::Tensor center(torch::Tensor  retv)
	{
		auto c1 = retv.select(1,0).unsqueeze(1);
		auto c2 = retv.select(1,1).unsqueeze(1);
		auto c3 = retv.select(1,2).unsqueeze(1);
		auto c4 = retv.select(1,3).unsqueeze(1);
	
		auto _retv = torch::cat({(c1+c3).div(2),(c2+c4).div(2),c3-c1,c4-c2},1);
		return _retv;

	}
	torch::Tensor decode(const torch::Tensor _loc,torch::Tensor _prior)
	{
		auto top_2 = torch::tensor({0,1}).cuda().to(torch::kLong);
		auto bottom_2 = torch::tensor({2,3}).cuda().to(torch::kLong);
		
		auto c1 = _prior.index_select(1,top_2)+_loc.index_select(1,top_2).mul(variance[0])*_prior.index_select(1,bottom_2);
		auto c2 = _prior.index_select(1,bottom_2)*torch::exp(_loc.index_select(1,bottom_2)*variance[1]);
		auto _retv = torch::cat({c1,c2},1);
		auto c3 = _retv.index_select(1,top_2)-_retv.index_select(1,bottom_2).div(2);
		auto c4 = c3 + _retv.index_select(1,bottom_2);
		return torch::cat({c3,c4},1);	

	}

};
void test(std::string &ok_img_file,std::string & test_img_file,std::string & model_path,std::string & csv_path)
{
	torch::NoGradGuard no_grad;
	string dir = "./JPEGImages/";
	string save_dir = "./result/";
	string _temp;
	string ok_dir = ok_img_file;
	string test_dir =test_img_file;
	
	std::ifstream fi(ok_dir);
	string ok_img_name;	
	std::vector<string> test_img_sets;
	std::vector<string> save_img_sets;
	Options options;
	
	while(std::getline(fi,_temp))
	{
		ok_img_name = dir+_temp+POSFIX;
	}
	fi.close();
	fi.open(test_dir);
	while(std::getline(fi,_temp))
	{
		test_img_sets.push_back(dir+_temp+POSFIX);
		save_img_sets.push_back(save_dir+_temp+POSFIX);
	}
	fi.close();
	auto OK = cv::imread(ok_img_name);	
	cv::resize(OK,OK,cv::Size(512,512));
	std::string priors_dir{csv_path};	
	torch::DeviceType device_type;
	if (torch::cuda::is_available() && !options.no_cuda) {
		std::cout << "CUDA available! Training on GPU" << std::endl;
		device_type = torch::kCUDA;
	} else {
		std::cout << "Training on CPU" << std::endl;
		device_type = torch::kCPU;
	}
	torch::Device device(device_type);
	std::shared_ptr<torch::jit::script::Module> model = torch::jit::load(model_path);

	std::vector<float> region_priors;
	forward_region_priors(priors_dir,region_priors);
	model->to(device);	
	auto priors = torch::from_blob(region_priors.data(),{16320,4}).cuda();

	std::cout<<"Successful input the model"<<std::endl;
	int iter = 0;
	for(auto test_name: test_img_sets)
	{
		std::cout<<"process:"<<test_name<<std::endl;
		auto orign = cv::imread(test_name);
		float h=orign.rows;
		float w = orign.cols;
		cv::Mat t;

		cv::resize(orign,t,cv::Size(512,512));
		torch::Tensor x1 = torch::from_blob(OK.data,{1,t.rows,t.cols,3},torch::kByte);
		x1 = x1.to(torch::kFloat).to(device);

		
		x1 = x1.permute({0,3,1,2});

		
		auto flag = torch::tensor(true).to(device);

		auto output = model->forward({x1,flag});
		auto tpl = output.toTuple();
		auto arm_loc = tpl->elements()[0].toTensor();
		auto arm_conf = tpl->elements()[1].toTensor();
		auto odm_loc = tpl->elements()[2].toTensor();
		auto odm_conf = tpl->elements()[3].toTensor();
		
		detector det(2,0.01);
		auto retv = det(odm_loc,odm_conf,priors,arm_loc,arm_conf);
		//get the rect and class confidence
#ifndef TEST	
		auto _boxes = std::get<0>(retv)[0];
		


		auto _scores = std::get<1>(retv)[0];
		float temp[] = {w,h,w,h};
		auto scale = torch::from_blob(temp,{4});
		_boxes = _boxes*scale;
		
		auto boxes = _boxes.accessor<float,2>();
		auto scores = _scores.accessor<float,2>();

		for(int i = 1;i<options.num_class;i++)
		{
			auto c_det = torch::zeros({boxes.size(0),5});
			auto class_scores = torch::zeros({boxes.size(0),1});
			size_t objs = 0;
			for(long j = 0;j<boxes.size(0);j++)
			{
				if(scores[j][i]>0.1)
				{
					for(int t =0;t<4;t++)
						c_det[objs][t] = boxes[j][t];						
					c_det[objs][4] = scores[j][i];
					class_scores[objs][0] = scores[j][i];
					objs++;
				}
			}

			//nms
			int num_out;
			std::vector<int> keep_out(boxes.size(0),0);
			auto sorted = nms_cuda(keep_out.data(),&num_out,c_det,class_scores,objs,0.5);

			//keep only top-200
			num_out = num_out>200?200:num_out;
			
			auto keep_det = torch::zeros({num_out,5});
			std::tuple<int,int> shapes(h,w);
			for(int j =0;j<num_out;j++)
			{
				if(c_det[j][4].item<float>() >0)
					keep_det[j] = c_det[sorted[keep_out[j]][0]];
			}
			//auto out = post_process(keep_det,shapes);
			auto out = keep_det;
			for(int j = 0;j<out.size(0);j++)
			{
				cv::rectangle(orign,cv::Point(out[j][0].item<int>(),out[j][1].item<int>()),cv::Point(out[j][2].item<int>(),out[j][3].item<int>()),cv::Scalar(0,0,255),3);
			}
			cv::imwrite(save_img_sets[iter++],orign);
			//cv::imshow("fuck",orign);
			//cv::waitKey();
		}
#endif
	}
}
auto main(int argc, const char * argv[])->int{
	assert(argc == 5);
	torch::manual_seed(0);
	
	auto x = torch::randn({3,5});

	std::string ok_txt{argv[1]};
	std::string ng_txt{argv[2]};
	std::string model_path{argv[4]};
	std::string csv_path{argv[3]};
	test(ok_txt,ng_txt,model_path,csv_path);
}
