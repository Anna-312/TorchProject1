#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
using namespace std;


int main() {
	//Загрузка изображения
	cv::Mat img = cv::imread(image_name);
	cv::Size target_size(28, 28);
	cv::resize(img, img, target_size);
	switch (img.channels())
	{
	case 4:
		cv::cvtColor(img, img, cv::COLOR_BGRA2RGB);
		break;
	case 3:
		//		cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
		cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);
		break;
	default:
		throw new std::runtime_error("incorrect image depth!");
	}
	torch::NoGradGuard no_grad;
	//Загрузка модели из файла
	torch::jit::script::Module module = torch::jit::load("mnist_final.pth");
	//Преобразование изображения в тензор
	torch::Tensor tensor_img = torch::from_blob(img.data, { 28, 28,1 }, torch::kByte);
	tensor_img = tensor_img.permute({ 2, 0, 1 }); // Channels x Height x Width
	//std::cout << tensor_img << "\n";
	tensor_img = tensor_img.to(torch::kF32);
	tensor_img = tensor_img / 255.0;
	//std::cout << tensor_img << "\n";
	//Получение и вывод результата работы нейросети
	at::Tensor output = module.forward({ tensor_img }).toTensor();
	int result = output.argmax().item().to<int>();
	cout << "Result: " << result;
	return 1;
}
