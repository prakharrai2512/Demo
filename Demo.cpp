#include "Demo.h"


int main() {
	torch::Tensor tensor = torch::rand({ 2, 3 }).to(at::kCUDA);
	std::cout << tensor << std::endl;
	std::cout << "Hello\n";
    torch::jit::script::Module module1;
    torch::jit::script::Module module2;
    torch::DeviceType device_type;
    device_type = torch::kCPU;
    std::cout << torch::cuda::is_available() << std::endl;

    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module1 = torch::jit::load("C:\\Users\\acer\\Desktop\\SA lab\\Demo\\data\\blaziken.pt");
        module1.to(device_type);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n"<<e.what();
        return -1;
    }

    torch::jit::script::Module data;

    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        data = torch::jit::load("C:\\Users\\acer\\Desktop\\SA lab\\Demo\\data\\data_blaze.pth");
        module1.to(device_type);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n" << e.what();
        return -1;
    }
    //data.to(device_type);
    auto img = data.attr("img").toTensor().to(device_type);
    auto val_det = data.attr("det_out").toTensor().to(device_type);
    auto da_det = data.attr("da_seg_out").toTensor().to(device_type);
    auto ll_det = data.attr("ll_seg_out").toTensor().to(device_type);

    std::cout << "Beginning inference\n";
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    /*auto det_out = module2.forward({img}).toTuple()->elements()[0].toTensor().to(at::kCUDA);

    auto maskout = module1.forward({ img }).toTensorVector();

    auto ll_out = maskout[1].to(at::kCUDA);
    auto da_out = maskout[0].to(at::kCUDA);
    */
    auto retTup = module1.forward({ img }).toTuple()->elements();
    auto da_out = retTup[0].toTensor().to(device_type);
    auto ll_out = retTup[1].toTensor().to(device_type);
    auto det_out = retTup[2].toTensor().to(device_type);

    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout<<"Elapsed time: " << elapsed_seconds.count() << "s\n";
    std::cout << "Iterations :" << 1 / elapsed_seconds.count() << " it/s\n";
    std::cout << "Done inference\n";

    if (det_out.equal(val_det)) {
        std::cout << "Detection same\n";
    }
    if (ll_out.equal(ll_det)) {
        std::cout << "Lane Line same\n";
    }
    if (da_out.equal(da_det)) {
        std::cout << "Driveable area same\n";
    }
    //std::cout << img<<std::endl;
    std::cout << "ok\n";


    return 0;
}