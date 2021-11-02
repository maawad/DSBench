#pragma once
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <unordered_set>

namespace DSBench {
namespace rkg {
template <typename key_type>
void generate_uniform_unique_keys(
    std::vector<key_type>& keys,
    std::size_t num_keys,
    bool cache = false,
    key_type sentinel_key = std::numeric_limits<key_type>::max(),
    int seed = 1) {
  keys.resize(num_keys);

  std::string dataset_dir = "dataset";
  std::string dataset_name = std::to_string(num_keys) + "_" + std::to_string(seed);
  std::string dataset_path = dataset_dir + "/" + dataset_name;
  if (cache) {
    if (std::filesystem::exists(dataset_dir)) {
      if (std::filesystem::exists(dataset_path)) {
        std::cout << "Reading cached keys.." << std::endl;
        std::ifstream dataset(dataset_path, std::ios::binary);
        dataset.read((char*)keys.data(), sizeof(key_type) * num_keys);
        dataset.close();
        return;
      }
    } else {
      std::filesystem::create_directory(dataset_dir);
    }
  }
  std::random_device rd;
  std::mt19937 rng(seed);
  auto min_key = std::numeric_limits<key_type>::min();
  auto max_key = std::numeric_limits<key_type>::max();
  std::uniform_int_distribution<key_type> uni(min_key, max_key);
  std::unordered_set<key_type> unique_keys;
  while (unique_keys.size() < num_keys) {
    auto key = uni(rng);
    if (key != sentinel_key) {
      unique_keys.insert(uni(rng));
    }
  }
  std::copy(unique_keys.cbegin(), unique_keys.cend(), keys.begin());
  std::shuffle(keys.begin(), keys.end(), rng);

  if (cache) {
    std::cout << "Caching.." << std::endl;
    std::ofstream dataset(dataset_path, std::ios::binary);
    dataset.write((char*)keys.data(), sizeof(key_type) * num_keys);
    dataset.close();
  }
}
}  // namespace rkg
}  // namespace DSBench