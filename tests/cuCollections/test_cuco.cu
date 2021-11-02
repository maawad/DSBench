#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <iostream>

#include <cuco/static_map.cuh>
#include <gpu_timer.hpp>

#include <cmd.hpp>

int main(int argc, char** argv) {
  auto arguments = std::vector<std::string>(argv, argv + argc);

  std::size_t num_keys =
      get_arg_value<std::size_t>(arguments, "num-keys").value_or(16ull);
  float load_factor = get_arg_value<float>(arguments, "load_factor").value_or(0.7f);
  std::size_t capacity = static_cast<std::size_t>(num_keys / load_factor);

  std::cout << "num-keys: " << num_keys << '\n';
  std::cout << "load-factor: " << load_factor << '\n';
  std::cout << "load-factor: " << load_factor << '\n';

  using key_type = uint32_t;
  using value_type = uint32_t;
  using pair_type = thrust::pair<key_type, value_type>;

  pair_type sentinel{0, 0};

  // generate keys
  thrust::device_vector<pair_type> d_pairs(num_keys);
  std::vector<pair_type> h_pairs(num_keys);

  thrust::transform(thrust::make_counting_iterator<int>(0),
                    thrust::make_counting_iterator<int>(d_pairs.size()),
                    d_pairs.begin(),
                    [] __device__(auto i) { return thrust::make_pair(i + 1, i + 1); });

  cuco::static_map<key_type, value_type> map{capacity, sentinel.first, sentinel.second};

  return 0;
}
