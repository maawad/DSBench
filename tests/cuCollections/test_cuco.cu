#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <cuda_helpers.cuh>


#include <filesystem>
#include <iostream>

#include <cuco/static_map.cuh>
#include <gpu_timer.hpp>
#include <rkg.hpp>


#include <cmd.hpp>


void output_rates(float insertion_seconds,
                          float find_seconds,
                          std::size_t num_insertions,
                          std::size_t num_finds,
                          float load_factor){

    std::cout << "Insertions: " << num_insertions << " keys" << '\n';
    double insertion_rate = double(num_insertions) * 1e-6 / insertion_seconds;
    std::cout << "Insertion_rate: " << insertion_rate << " Mkey/s";
    std::cout << " (" << insertion_seconds << " seconds)" << "\n\n";
    std::cout << "Finds: " << num_finds << " keys" << '\n';
    double find_rate = double(num_finds) * 1e-6 / find_seconds;
    std::cout << "find_rate: " << find_rate << " Mkey/s" ;
    std::cout << " (" << find_seconds << " seconds)" << "\n\n";

    // print to file
    std::string result_dir = "./";
    std::string result_fname = "cuco.csv";

    bool output_file_exist = std::filesystem::exists(result_dir + result_fname);

    std::fstream output(result_dir + result_fname, std::ios::app);
    if (!output_file_exist) {
      // header
      output << "num_insertions, num_finds, load_factor,";
      output << "insert,";
      output << "find,\n";
    }

  output << num_insertions << ",";
  output << num_finds << ",";
  output << load_factor << ",";
  output << insertion_rate << ",";
  output << find_rate << ",\n";
}
int main(int argc, char** argv) {
  auto arguments = std::vector<std::string>(argv, argv + argc);

  std::size_t num_keys =
      get_arg_value<std::size_t>(arguments, "num-keys").value_or(16ull);
  float load_factor = get_arg_value<float>(arguments, "load-factor").value_or(0.7f);
  int device_id = get_arg_value<int>(arguments, "device").value_or(0);

  DSBench::set_device(device_id);

  std::size_t capacity = static_cast<std::size_t>(num_keys / load_factor);

  std::cout << "num-keys: " << num_keys << '\n';
  std::cout << "capacity: " << capacity << '\n';
  std::cout << "load-factor: " << load_factor << "\n\n";

  using key_type = uint32_t;
  using value_type = uint32_t;
  using pair_type = thrust::pair<key_type, value_type>;
  using map_type = cuco::static_map<key_type, value_type>;
  pair_type sentinel{0, 0};

 auto key_to_pair = [] __host__ __device__(key_type x) { return pair_type{x,x % 512}; };

  // generate keys
  thrust::device_vector<key_type> d_keys(num_keys);
  std::vector<key_type> h_keys(num_keys);
  thrust::device_vector<pair_type> d_pairs(num_keys);


  // Generate keys
  DSBench::rkg::generate_uniform_unique_keys(h_keys, num_keys, true, sentinel.first, 1);
  d_keys = h_keys;

  // keys to pairs
  thrust::transform(
      thrust::device, d_keys.begin(), d_keys.end(), d_pairs.begin(), key_to_pair);

  // Queries
  thrust::device_vector<key_type> d_queries(d_keys);
  thrust::device_vector<value_type> d_results(num_keys, sentinel.second);

  map_type map{capacity, sentinel.first, sentinel.second};

  // Insertions
  gpu_timer insertion_timer;
  insertion_timer.start_timer();
  map.insert(d_pairs.begin(), d_pairs.end());
  insertion_timer.stop_timer();
  auto insertion_seconds = insertion_timer.get_elapsed_s();

  // finds
  gpu_timer find_timer;
  find_timer.start_timer();
  map.find(d_keys.begin(), d_keys.end(), d_results.begin());
  find_timer.stop_timer();
  auto find_seconds = find_timer.get_elapsed_s();

  thrust::host_vector<key_type> h_queries(d_queries);
  thrust::host_vector<value_type> h_results(d_results);

  output_rates(insertion_seconds, find_seconds, num_keys, num_keys, load_factor);

  for (std::size_t i = 0; i < num_keys; i++) {
    auto key = h_queries[i];
    auto expected_pair = key_to_pair(key);
    auto found_result = h_results[i];
    if (expected_pair.second!= found_result) {
      std::cout << "Error: expected: " <<expected_pair.second;
      std::cout << ", found: " << found_result << '\n';
      return 1;
    }
  }
  std::cout << "Success\n";



  return 0;

}
