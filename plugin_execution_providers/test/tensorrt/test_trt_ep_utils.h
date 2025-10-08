#include <string>
#include <vector>

namespace test {
namespace trt_ep {
std::string ToUTF8String(std::wstring_view s);
std::wstring ToWideString(std::string_view s);

void CreateBaseModel(const std::string& model_path,
                     const std::string& graph_name,
                     const std::vector<int64_t>& dims,
                     bool add_non_zero_node = false);
}
}