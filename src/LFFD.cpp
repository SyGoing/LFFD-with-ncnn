
#include "LFFD.h"

LFFD::LFFD(const std::string& model_path, int scale_num, int num_thread_)
{
	num_output_scales = scale_num;
	num_thread = num_thread_;
	if (num_output_scales == 5) {
		param_file_name = model_path+ "/symbol_10_320_20L_5scales_v2_deploy.param";
		bin_file_name = model_path+"/train_10_320_20L_5scales_v2_iter_1800000.bin";
		receptive_field_list = { 20, 40, 80, 160, 320 };
		receptive_field_stride = { 4, 8, 16, 32, 64 };
		bbox_small_list = { 10, 20, 40, 80, 160 };
		bbox_large_list = { 20, 40, 80, 160, 320 };
		receptive_field_center_start = { 3, 7, 15, 31, 63 };

		for (int i = 0; i < receptive_field_list.size(); i++) {
			constant.push_back(receptive_field_list[i] / 2);
		}

		output_blob_names = { "softmax0","conv8_3_bbox",
		                                  "softmax1","conv11_3_bbox",
										  "softmax2","conv14_3_bbox",
		                                  "softmax3","conv17_3_bbox",
		                                  "softmax4","conv20_3_bbox" };

	}
	else if (num_output_scales == 8) {
		param_file_name = model_path+"/symbol_10_560_25L_8scales_v1_deploy.param";
		bin_file_name = model_path+"/train_10_560_25L_8scales_v1_iter_1400000.bin";
		receptive_field_list = { 15, 20, 40, 70, 110, 250, 400, 560 };
		receptive_field_stride = { 4, 4, 8, 8, 16, 32, 32, 32 };
		bbox_small_list = { 10, 15, 20, 40, 70, 110, 250, 400 };
		bbox_large_list = { 15, 20, 40, 70, 110, 250, 400, 560 };
		receptive_field_center_start = { 3, 3, 7, 7, 15, 31, 31, 31 };

		for (int i = 0; i < receptive_field_list.size(); i++) {
			constant.push_back(receptive_field_list[i] / 2);
		}

		output_blob_names = { "softmax0","conv8_3_bbox",
			"softmax1","conv10_3_bbox",
			"softmax2","conv13_3_bbox",
			"softmax3","conv15_3_bbox",
			"softmax4","conv18_3_bbox",
			"softmax5","conv21_3_bbox",
			"softmax6","conv23_3_bbox",
		    "softmax7","conv25_3_bbox" };
	}

	lffd.load_param(param_file_name.data());
	lffd.load_model(bin_file_name.data());

}

LFFD::~LFFD()
{
	lffd.clear();
}

int LFFD::detect(ncnn::Mat& img, std::vector<FaceInfo>& face_list, int resize_h, int resize_w,
	float score_threshold, float nms_threshold, int top_k, std::vector<int> skip_scale_branch_list)
{

	if (img.empty()) {
		std::cout << "image is empty ,please check!" << std::endl;
		return -1;
	}

	image_h = img.h;
	image_w = img.w;

    ncnn::Mat in;
    ncnn::resize_bilinear(img,in,resize_w,resize_h);
    float ratio_w=(float)image_w/in.w;
    float ratio_h=(float)image_h/in.h;

	ncnn::Mat ncnn_img = in;
	ncnn_img.substract_mean_normalize(mean_vals, norm_vals);

	std::vector<FaceInfo> bbox_collection;
	ncnn::Extractor ex = lffd.create_extractor();
	ex.set_num_threads(num_thread);
	ex.input("data", ncnn_img);

	for (int i = 0; i <num_output_scales; i++) {
		ncnn::Mat conf;
		ncnn::Mat reg;

		ex.extract(output_blob_names[2*i].c_str(), conf);
		ex.extract(output_blob_names[2 * i+1].c_str(), reg);

		generateBBox(bbox_collection, conf, reg, score_threshold, conf.w, conf.h, in.w, in.h, i);
	}
	std::vector<FaceInfo> valid_input;
	get_topk_bbox(bbox_collection, valid_input, top_k);
	nms(valid_input, face_list, nms_threshold);

    for(int i=0;i<face_list.size();i++){
        face_list[i].x1*=ratio_w;
        face_list[i].y1*=ratio_h;
        face_list[i].x2*=ratio_w;
        face_list[i].y2*=ratio_h;

        float w,h,maxSize;
        float cenx,ceny;
        w=face_list[i].x2-face_list[i].x1;
        h=face_list[i].y2-face_list[i].y1;

		maxSize = w > h ? w : h;
        cenx=face_list[i].x1+w/2;
        ceny=face_list[i].y1+h/2;
        face_list[i].x1=cenx-maxSize/2>0? cenx - maxSize / 2:0;
        face_list[i].y1=ceny-maxSize/2>0? ceny - maxSize / 2:0;
        face_list[i].x2=cenx+maxSize/2>image_w? image_w-1: cenx + maxSize / 2;
        face_list[i].y2=ceny+maxSize/2> image_h? image_h-1: ceny + maxSize / 2;

    }
	return 0;
}

void LFFD::generateBBox(std::vector<FaceInfo>& bbox_collection, ncnn::Mat score_map, ncnn::Mat box_map, float score_threshold, int fea_w, int fea_h, int cols, int rows, int scale_id)
{
	float* RF_center_Xs = new float[fea_w];
	float* RF_center_Xs_mat = new float[fea_w * fea_h];
	float* RF_center_Ys = new float[fea_h];
	float* RF_center_Ys_mat = new float[fea_h * fea_w];

    for (int x = 0; x < fea_w; x++) {
		RF_center_Xs[x] = receptive_field_center_start[scale_id] + receptive_field_stride[scale_id] * x;
	}
	for (int x = 0; x < fea_h; x++) {
		for (int y = 0; y < fea_w; y++) {
			RF_center_Xs_mat[x * fea_w + y] = RF_center_Xs[y];
		}
	}

	for (int x = 0; x < fea_h; x++) {
		RF_center_Ys[x] = receptive_field_center_start[scale_id] + receptive_field_stride[scale_id] * x;
		for (int y = 0; y < fea_w; y++) {
			RF_center_Ys_mat[x * fea_w + y] = RF_center_Ys[x];
		}
	}

	float* x_lt_mat = new float[fea_h * fea_w];
	float* y_lt_mat = new float[fea_h * fea_w];
	float* x_rb_mat = new float[fea_h * fea_w];
	float* y_rb_mat = new float[fea_h * fea_w];

	

	//x-left-top
	float mid_value = 0;
	for (int j = 0; j < fea_h * fea_w; j++) {
		mid_value = RF_center_Xs_mat[j] - box_map.channel(0)[j] * constant[scale_id];
		x_lt_mat[j] = mid_value < 0 ? 0 : mid_value;
	}
	//y-left-top
	for (int j = 0; j < fea_h * fea_w; j++) {
		mid_value = RF_center_Ys_mat[j] - box_map.channel(1)[j] * constant[scale_id];
		y_lt_mat[j] = mid_value < 0 ? 0 : mid_value;
	}
	//x-right-bottom
	for (int j = 0; j < fea_h * fea_w; j++) {
		mid_value = RF_center_Xs_mat[j] - box_map.channel(2)[j] * constant[scale_id];
		x_rb_mat[j] = mid_value > cols - 1 ? cols - 1 : mid_value;
	}
	//y-right-bottom
	for (int j = 0; j < fea_h * fea_w; j++) {
		mid_value = RF_center_Ys_mat[j] - box_map.channel(3)[j] * constant[scale_id];
		y_rb_mat[j] = mid_value > rows - 1 ? rows - 1 : mid_value;
	}

	for (int k = 0; k < fea_h * fea_w; k++) {
		if (score_map.channel(0)[k] > score_threshold) {
			FaceInfo faceinfo;
			faceinfo.x1 = x_lt_mat[k];
			faceinfo.y1 = y_lt_mat[k];
			faceinfo.x2 = x_rb_mat[k];
			faceinfo.y2 = y_rb_mat[k];
			faceinfo.score = score_map[k];
			faceinfo.area = (faceinfo.x2 - faceinfo.x1) * (faceinfo.y2 - faceinfo.y1);
			bbox_collection.push_back(faceinfo);
		}
	}

	delete[] RF_center_Xs; RF_center_Xs = NULL;
	delete[] RF_center_Ys; RF_center_Ys = NULL;
	delete[] RF_center_Xs_mat; RF_center_Xs_mat = NULL;
	delete[] RF_center_Ys_mat; RF_center_Ys_mat = NULL;
	delete[] x_lt_mat; x_lt_mat = NULL;
	delete[] y_lt_mat; y_lt_mat = NULL;
	delete[] x_rb_mat; x_rb_mat = NULL;
	delete[] y_rb_mat; y_rb_mat = NULL;
}

void LFFD::get_topk_bbox(std::vector<FaceInfo>& input, std::vector<FaceInfo>& output, int top_k)
{
	std::sort(input.begin(), input.end(),
		[](const FaceInfo& a, const FaceInfo& b)
		{
			return a.score > b.score;
		});

	if (input.size() > top_k) {
		for (int k = 0; k < top_k; k++) {
			output.push_back(input[k]);
		}
	}
	else {
		output = input;
	}
}

void LFFD::nms(std::vector<FaceInfo>& input, std::vector<FaceInfo>& output, float threshold, int type)
{
	if (input.empty()) {
		return;
	}
	std::sort(input.begin(), input.end(),
		[](const FaceInfo& a, const FaceInfo& b)
		{
			return a.score < b.score;
		});

	float IOU = 0;
	float maxX = 0;
	float maxY = 0;
	float minX = 0;
	float minY = 0;
	std::vector<int> vPick;
	int nPick = 0;
	std::multimap<float, int> vScores;
	const int num_boxes = input.size();
	vPick.resize(num_boxes);
	for (int i = 0; i < num_boxes; ++i) {
		vScores.insert(std::pair<float, int>(input[i].score, i));
	}
	while (vScores.size() > 0) {
		int last = vScores.rbegin()->second;
		vPick[nPick] = last;
		nPick += 1;
		for (std::multimap<float, int>::iterator it = vScores.begin(); it != vScores.end();) {
			int it_idx = it->second;
			maxX = max(input.at(it_idx).x1, input.at(last).x1);
			maxY = max(input.at(it_idx).y1, input.at(last).y1);
			minX = min(input.at(it_idx).x2, input.at(last).x2);
			minY = min(input.at(it_idx).y2, input.at(last).y2);
			//maxX1 and maxY1 reuse 
			maxX = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
			maxY = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;
			//IOU reuse for the area of two bbox
			IOU = maxX * maxY;
			if (type==NMS_UNION)
				IOU = IOU / (input.at(it_idx).area + input.at(last).area - IOU);
			else if (type == NMS_MIN) {
				IOU = IOU / ((input.at(it_idx).area < input.at(last).area) ? input.at(it_idx).area : input.at(last).area);
			}
			if (IOU > threshold) {
				it = vScores.erase(it);
			}
			else {
				it++;
			}
		}
	}

	vPick.resize(nPick);
	output.resize(nPick);
	for (int i = 0; i < nPick; i++) {
		output[i] = input[vPick[i]];
	}
}
