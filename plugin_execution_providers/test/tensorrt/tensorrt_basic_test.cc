#include <gtest/gtest.h>
#include <string>
#include <gsl/gsl>

#include "onnxruntime_cxx_api.h"
#include "test_trt_ep_utils.h"
#include "path_string.h"

namespace test {
namespace trt_ep {
// char type for filesystem paths
using PathChar = ORTCHAR_T;
// string type for filesystem paths
using PathString = std::basic_string<PathChar>;

class TensorrtExecutionProviderCacheTest : public testing::TestWithParam<std::string> {};

OrtStatus* CreateOrtSession(PathString model_name,
                            std::string lib_registration_name,
                            PathString lib_path) {
  const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  Ort::Env env;

  // Register plugin TRT EP library with ONNX Runtime.
  env.RegisterExecutionProviderLibrary(
      lib_registration_name.c_str(),  // Registration name can be anything the application chooses.
      lib_path                        // Path to the plugin TRT EP library.
  );

  // Unregister the library using the application-specified registration name.
  // Must only unregister a library after all sessions that use the library have been released.
  auto unregister_plugin_eps_at_scope_exit =
      gsl::finally([&]() { env.UnregisterExecutionProviderLibrary(lib_registration_name.c_str()); });

  {
    std::vector<Ort::ConstEpDevice> ep_devices = env.GetEpDevices();
    // EP name should match the name assigned by the EP factory when creating the EP (i.e., in the implementation of
    // OrtEP::CreateEp())
    std::string ep_name = lib_registration_name;

    // Find the Ort::EpDevice for "TensorRTEp".
    std::vector<Ort::ConstEpDevice> selected_ep_devices = {};
    for (Ort::ConstEpDevice ep_device : ep_devices) {
      if (std::string(ep_device.EpName()) == ep_name) {
        selected_ep_devices.push_back(ep_device);
        break;
      }
    }

    if (selected_ep_devices[0] == nullptr) {
      // Did not find EP. Report application error ...
      std::string message = "Did not find EP: " + ep_name;
      return ort_api->CreateStatus(ORT_EP_FAIL, message.c_str());
    }

    std::unordered_map<std::string, std::string> ep_options;  // Optional EP options.
    Ort::SessionOptions session_options;
    session_options.AppendExecutionProvider_V2(env, selected_ep_devices, ep_options);

    Ort::Session session(env, model_name.c_str(), session_options);

    // Get default ORT allocator
    Ort::AllocatorWithDefaultOptions allocator;

    // Get input name
    Ort::AllocatedStringPtr input_name_ptr =
        session.GetInputNameAllocated(0, allocator);  // Keep the smart pointer alive to avoid dangling pointer
    const char* input_name = input_name_ptr.get();

  }


}

TEST(TensorrtExecutionProviderTest, SessionCreationWithMultiThreadsAndInferenceWithMultiThreads) {
  std::vector<std::thread> threads;
  std::string model_name = "basic_model_for_test.onnx";
  std::string graph_name = "basic_model";
  std::string lib_registration_name = "TensorRTEp";
  PathString lib_path = ORT_TSTR("TensorRTEp.dll");
  std::vector<int64_t> dims = {1, 3, 2};
  CreateBaseModel(model_name, graph_name, dims);
  CreateOrtSession(ToPathString(model_name), lib_registration_name, lib_path);
}

}  // namespace trt_ep
}  // namespace test