#ifndef __FLAT_HASH_TABLE_H__
#define __FLAT_HASH_TABLE_H__

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hash_table_helpers.h"

using namespace std;

namespace falconn {
namespace core {

class FlatHashTableError : public HashTableError {
 public:
  FlatHashTableError(const char* msg) : HashTableError(msg) {}
};

template <typename KeyType, typename ValueType = int32_t,
          typename IndexType = int32_t>
class FlatHashTable {
 public:
  class Factory {
   public:
    Factory(IndexType num_buckets) : num_buckets_(num_buckets) {
      if (num_buckets_ < 1) {
        throw FlatHashTableError("Number of buckets must be at least 1.");
      }
    }

    FlatHashTable<KeyType, ValueType, IndexType>* new_hash_table() {
      return new FlatHashTable<KeyType, ValueType, IndexType>(num_buckets_);
    }

   private:
    IndexType num_buckets_ = 0;
  };

  typedef IndexType* Iterator;

  FlatHashTable(IndexType num_buckets) : num_buckets_(num_buckets) {}

  // TODO: add version with explicit values array? (maybe not because the flat
  // hash table is arguably most useful for the static table setting?)
  void add_entries(const std::vector<KeyType>& keys) {
    if (num_buckets_ <= 0) {
      throw FlatHashTableError("Non-positive number of buckets");
    }
    if (entries_added_) {
      throw FlatHashTableError("Entries were already added.");
    }
    bucket_list_.resize(num_buckets_, std::make_pair(0, 0));

    entries_added_ = true;

    KeyComparator comp(keys);
    std::vector<ValueType> indices_temp;
    indices_temp.resize(keys.size());
    for (IndexType ii = 0; static_cast<size_t>(ii) < indices_temp.size(); ++ii) {
      if (keys[ii] >= static_cast<KeyType>(num_buckets_) || keys[ii] < 0) {
        throw FlatHashTableError("Key value out of range.");
      }
      indices_temp[ii] = ii;
    }
    std::sort(indices_temp.begin(), indices_temp.end(), comp);

    IndexType cur_index = 0;
    indices_.resize(indices_temp.size() * 2);
    IndexType cur_temp_index = 0;
    while (cur_temp_index < static_cast<IndexType>(indices_temp.size())) {
      IndexType end_index = cur_index;
      IndexType end_temp_index = cur_temp_index;

      do {
        this->fill_indices(end_index, indices_temp[end_temp_index]);
        end_index++;
        end_temp_index++;
      } while (end_temp_index < static_cast<IndexType>(indices_temp.size()) &&
               keys[indices_temp[cur_temp_index]] == keys[indices_temp[end_temp_index]]);

      // 在每个bucket预留一些位置，为insert数据节省时间
      int_fast16_t reserved_size = this->get_reserved_size(end_index - cur_index);
      IndexType valid_end_index = end_index;
      for (int_fast8_t j = 0; j < reserved_size; j++) {
        this->fill_indices(end_index, -1);
        end_index++;
      }

      bucket_list_[keys[indices_[cur_index]]].first = cur_index;
      bucket_list_[keys[indices_[cur_index]]].second = valid_end_index - cur_index;
      cur_index = end_index;
      cur_temp_index = end_temp_index;
    }

    indices_.resize(cur_index);
  }

  void insert(IndexType key, int_fast64_t dataset_size) {
    // 如果该bucket中没有数据
    size_t index = -1;
    if (bucket_list_.at(key).second == 0) {
        IndexType valid_bucket_first = -1;
        // 向后查找
        for (uint_fast32_t i = key + 1; i < bucket_list_.size(); i++) {
            if (bucket_list_.at(i).first != 0) {
                valid_bucket_first = bucket_list_.at(i).first;
                break;
            }
        }

        // 如果后面的bucket全部没有数据，向前查找
        if (valid_bucket_first == -1) {
            for (int_fast32_t i = key - 1; i < 0; i--) {
                if (bucket_list_.at(i).first != 0) {
                    valid_bucket_first = indices_.size();
//                            bucket_list_.at(i).first + bucket_list_.at(i).second;                    break;
                }
            }
        }

        bucket_list_.at(key).first = valid_bucket_first;
        index = valid_bucket_first;
    } else {
        index = bucket_list_.at(key).first + bucket_list_.at(key).second;
    }

    // 存在预留位置
    if (index < indices_.size() && indices_.at(index) == -1) {
        indices_.at(index) = dataset_size;
    } else {
        indices_.insert(indices_.begin() + index, dataset_size);
        int_fast16_t reserved_size = this->get_reserved_size(bucket_list_.at(key).second);
        for (int_fast16_t i = 0; i < reserved_size; i++) {
            indices_.insert(indices_.begin() + index + i + 1, -1);
        }

        for (uint_fast32_t i = key + 1; i < bucket_list_.size(); i++) {
            if (bucket_list_.at(i).second > 0) {
              bucket_list_.at(i).first += 1 + reserved_size;
            }
        }
    }

    bucket_list_.at(key).second += 1;
  }

  void remove(ValueType point_index) {
       // 找出index在indices的位置
//      int indices_index = indices_.at(point_index);
      int indices_index = std::find(indices_.begin(), indices_.end(), point_index) - indices_.begin();

//      indices_.erase(indices_.begin() + indices_index);
      uint_fast32_t cur_bucket_index = 0;
      KeyType next_bucket_first = 0;
      for (uint_fast32_t bucket_index = 0; bucket_index < bucket_list_.size(); bucket_index++)
      {
          // 找出要删除的index在哪个bucket_list中
          if (indices_index < bucket_list_.at(bucket_index).first) {
              next_bucket_first = bucket_list_.at(bucket_index).first;
              break;
          }

          if (bucket_list_.at(bucket_index).first != 0) {
              cur_bucket_index = bucket_index;
          }
      }



//      KeyType next_bucket_first = 0;
//      for (KeyType i = bucket_index + 1; i < bucket_list_.size(); i++) {
//          if (bucket_list_.at(i).first != 0) {
//              next_bucket_first = bucket_list_.at(i).first;
//              break;
//          }
//      }
      if (next_bucket_first == 0) {
          next_bucket_first = indices_.size();
      }

      // 该bucket中后面的值往前挪
      for (KeyType i = indices_index; i < next_bucket_first; i++) {
          indices_.at(i) = indices_.at(i+1);
      }
      indices_.at(next_bucket_first-1) = -1;

      bucket_list_.at(cur_bucket_index).second -= 1;
      if (bucket_list_.at(cur_bucket_index).second == 0) {
          bucket_list_.at(cur_bucket_index).first = 0;
      }

//      for (uint_fast32_t i = bucket_index + 1; i < bucket_list_.size(); i++) {
//          if (bucket_list_.at(i).second > 0) {
//                bucket_list_.at(i).first -= 1;
//          }
//      }
  }

  void add_entries_new(const std::vector<KeyType>& keys) {
    if (num_buckets_ <= 0) {
      throw FlatHashTableError("Non-positive number of buckets");
    }
    if (entries_added_) {
      throw FlatHashTableError("Entries were already added.");
    }
    bucket_list_.resize(num_buckets_, std::make_pair(0, 0));

    entries_added_ = true;

    KeyComparator comp(keys);
    indices_.resize(keys.size());
    for (IndexType ii = 0; static_cast<size_t>(ii) < indices_.size(); ++ii) {
      if (keys[ii] >= static_cast<KeyType>(num_buckets_) || keys[ii] < 0) {
        throw FlatHashTableError("Key value out of range.");
      }
      indices_[ii] = ii;
    }
    std::sort(indices_.begin(), indices_.end(), comp);

    IndexType cur_index = 0;
    while (cur_index < static_cast<IndexType>(indices_.size())) {
      IndexType end_index = cur_index;
      do {
        end_index += 1;
      } while (end_index < static_cast<IndexType>(indices_.size()) &&
               keys[indices_[cur_index]] == keys[indices_[end_index]]);

      bucket_list_[keys[indices_[cur_index]]].first = cur_index;
      bucket_list_[keys[indices_[cur_index]]].second = end_index - cur_index;
      cur_index = end_index;
    }
  }

  std::pair<Iterator, Iterator> retrieve(const KeyType& key) {
    IndexType start = bucket_list_[key].first;
    IndexType len = bucket_list_[key].second;
    // printf("retrieve for key %u\n", key);
    // printf("  start: %lld  len %lld\n", start, len);
    return std::make_pair(&(indices_[start]), &(indices_[start + len]));
  }

 private:
        int_fast16_t get_reserved_size(int bucket_size) {
        int_fast16_t deserved_size = int_fast16_t(bucket_size/5);
        int_fast16_t reserved_size = deserved_size > 3 ? deserved_size : 3;
        return reserved_size;
    }

    void fill_indices(IndexType index, ValueType value) {
        if (indices_.size() <= index) {
            indices_.resize(index * 2);
        }
        indices_[index] = value;
    }


 private:
  IndexType num_buckets_ = -1;
  int_fast16_t min_reserved_pos_ = 3;
  bool entries_added_ = false;
  // the pair contains start index and length
  std::vector<std::pair<IndexType, IndexType>> bucket_list_;
  // point indices
  std::vector<ValueType> indices_;

  class KeyComparator {
   public:
    KeyComparator(const std::vector<KeyType>& keys) : keys_(keys) {}

    bool operator()(IndexType ii, IndexType jj) {
      return keys_[ii] < keys_[jj];
    }

    const std::vector<KeyType>& keys_;
  };
};

}  // namespace core
}  // namespace falconn

#endif
