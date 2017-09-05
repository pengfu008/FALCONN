#include <iostream>
#include <falconn/falconn_global.h>
#include <falconn/lsh_nn_table.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>

#include <memory>

namespace falconn {
namespace python {

class PyLSHNearestNeighborTableError : public FalconnError {
 public:
  PyLSHNearestNeighborTableError(const char *msg) : FalconnError(msg) {}
};

namespace py = pybind11;

template <typename T>
using NumPyArray = py::array_t<T, py::array::c_style>;
//template <typename T>
//using EigenMap = Eigen::Map<DenseVector<T>>;

//template <typename T>
//using Vec = DenseVector<T>;
//template <typename T>
//using PointSet = std::vector<Vec>;

//template <typename T>
//inline EigenMap<T> numpy_to_eigen(NumPyArray<T> x) {
//  py::buffer_info buf = x.request();
//  if (buf.ndim != 1) {
//    throw PyLSHNearestNeighborTableError("expected a one-dimensional array");
//  }
//  return EigenMap<T>((T *)buf.ptr, buf.shape[0]);
//}

//<template T>
//PointSet<T> numpy_to_array_dataset(NumPyArray<T> dataset) {
//  py::buffer_info buf = dataset.request();
//  if (buf.ndim != 2) {
//    throw PyLSHNearestNeighborTableError("expected a two-dimensional array");
//  }
//  size_t num_points = buf.shape[0];
//  size_t dimension = buf.shape[1];
//
//  PointSet<T> converted_points;
//  for (int_fast32_t i = 0; i < num_points; i++) {
//      Vec v(dimension);
//      for (int_fast32_t j = 0; j < dimension; j++) {
//         v[j] = ((T *)buf.ptr)[i*dimension + j];
//      }
//      converted_points.push_back(v);
//  }
//  return converted_points;
//}

template <typename T>
using LSHTable = LSHNearestNeighborTable<DenseVector<T>>;
template <typename T>
using LSHQueryObject = LSHNearestNeighborQuery<DenseVector<T>>;
template <typename T>
using LSHQueryPool = LSHNearestNeighborQueryPool<DenseVector<T>>;

namespace single_precision {

typedef float ScalarType;
typedef DenseVector<ScalarType> InnerVector;
typedef LSHTable<ScalarType> InnerLSHTable;
typedef LSHQueryObject<ScalarType> InnerLSHQueryObject;
typedef LSHQueryPool<ScalarType> InnerLSHQueryPool;

typedef NumPyArray<ScalarType> OuterNumPyArray;


typedef std::vector<InnerVector> InnerPointSet;
inline InnerPointSet numpy_to_array_dataset(OuterNumPyArray dataset) {
  py::buffer_info buf = dataset.request();
  if (buf.ndim != 2) {
    throw PyLSHNearestNeighborTableError("expected a two-dimensional array");
  }
  size_t num_points = buf.shape[0];
  size_t dimension = buf.shape[1];

  InnerPointSet converted_points;
  for (size_t i = 0; i < num_points; i++) {
      InnerVector v(dimension);
      for (size_t j = 0; j < dimension; j++) {
         v[j] = ((ScalarType *)buf.ptr)[i*dimension + j];
      }
      converted_points.push_back(v);
  }

//  free((ScalarType *)buf.ptr);
//  buf.ptr = NULL;

  return converted_points;
}

inline InnerVector numpy_to_array_point(OuterNumPyArray point) {
  py::buffer_info buf = point.request();
  if (buf.ndim != 1) {
    throw PyLSHNearestNeighborTableError("expected a two-dimensional array");
  }

  InnerVector converted_point(buf.shape[0]);
  for (int_fast32_t i = 0; i < buf.shape[0]; i++) {
     converted_point[i] = ((ScalarType *)buf.ptr)[i];
  }

  return converted_point;
}

//inline EigenMap<T> numpy_to_eigen(NumPyArray<T> x) {
//  py::buffer_info buf = x.request();
//  if (buf.ndim != 1) {
//    throw PyLSHNearestNeighborTableError("expected a one-dimensional array");
//  }
//  return EigenMap<T>((T *)buf.ptr, buf.shape[0]);
//}


class PyLSHNearestNeighborQueryDenseFloat {
 public:
  PyLSHNearestNeighborQueryDenseFloat(
      std::shared_ptr<InnerLSHQueryObject> query_object)
      : inner_entity_(query_object) {}

  void set_num_probes(int_fast64_t num_probes) {
    py::gil_scoped_release release;
    inner_entity_->set_num_probes(num_probes);
  }

  int_fast64_t get_num_probes() {
    py::gil_scoped_release release;
    return inner_entity_->get_num_probes();
  }

  void set_max_num_candidates(int_fast64_t max_num_candidates) {
    py::gil_scoped_release release;
    return inner_entity_->set_max_num_candidates(max_num_candidates);
  }

  int_fast64_t get_max_num_candidates() {
    py::gil_scoped_release release;
    return inner_entity_->get_max_num_candidates();
  }

  int32_t find_nearest_neighbor(OuterNumPyArray q) {
    InnerVector converted_query = numpy_to_array_point(q);
    py::gil_scoped_release release;
    return inner_entity_->find_nearest_neighbor(converted_query);
  }

  std::vector<std::pair<int32_t, float>> find_k_nearest_neighbors(OuterNumPyArray q,
                                                int_fast64_t k, float threshold) {
    InnerVector converted_query = numpy_to_array_point(q);

    py::gil_scoped_release release;
    std::vector<std::pair<int32_t, float>> result;
    inner_entity_->find_k_nearest_neighbors(converted_query, k, threshold, &result);
    return result;
  }

  std::vector<int32_t> find_near_neighbors(OuterNumPyArray q,
                                           ScalarType threshold) {
    InnerVector converted_query = numpy_to_array_point(q);
    py::gil_scoped_release release;
    std::vector<int32_t> result;
    inner_entity_->find_near_neighbors(converted_query, threshold, &result);
    return result;
  }

  std::vector<int32_t> get_unique_candidates(OuterNumPyArray q) {
    InnerVector converted_query = numpy_to_array_point(q);
    py::gil_scoped_release release;
    std::vector<int32_t> result;
    inner_entity_->get_unique_candidates(converted_query, &result);
    return result;
  }

  std::vector<int32_t> get_candidates_with_duplicates(OuterNumPyArray q) {
    InnerVector converted_query = numpy_to_array_point(q);
    py::gil_scoped_release release;
    std::vector<int32_t> result;
    inner_entity_->get_candidates_with_duplicates(converted_query, &result);
    return result;
  }

  void reset_query_statistics() {
    py::gil_scoped_release release;
    inner_entity_->reset_query_statistics();
  }

  QueryStatistics get_query_statistics() {
    py::gil_scoped_release release;
    return inner_entity_->get_query_statistics();
  }

 private:
  std::shared_ptr<InnerLSHQueryObject> inner_entity_;
};

class PyLSHNearestNeighborQueryPoolDenseFloat {
 public:
  PyLSHNearestNeighborQueryPoolDenseFloat(
      std::shared_ptr<InnerLSHQueryPool> query_pool)
      : inner_entity_(query_pool) {}

  void set_num_probes(int_fast64_t num_probes) {
    py::gil_scoped_release release;
    inner_entity_->set_num_probes(num_probes);
  }

  int_fast64_t get_num_probes() {
    py::gil_scoped_release release;
    return inner_entity_->get_num_probes();
  }

  void set_max_num_candidates(int_fast64_t max_num_candidates) {
    py::gil_scoped_release release;
    return inner_entity_->set_max_num_candidates(max_num_candidates);
  }

  int_fast64_t get_max_num_candidates() {
    py::gil_scoped_release release;
    return inner_entity_->get_max_num_candidates();
  }

  int32_t find_nearest_neighbor(OuterNumPyArray q) {
    InnerVector converted_query = numpy_to_array_point(q);
    py::gil_scoped_release release;
    return inner_entity_->find_nearest_neighbor(converted_query);
  }

  std::vector<std::pair<int32_t, float>> find_k_nearest_neighbors(OuterNumPyArray q,
                                                int_fast64_t k, float threshold) {
    InnerVector converted_query = numpy_to_array_point(q);
    py::gil_scoped_release release;
    std::vector<std::pair<int32_t, float>> result;
    inner_entity_->find_k_nearest_neighbors(converted_query, k, threshold, &result);
    return result;
  }

  std::vector<int32_t> find_near_neighbors(OuterNumPyArray q,
                                           ScalarType threshold) {
    InnerVector converted_query = numpy_to_array_point(q);
    py::gil_scoped_release release;
    std::vector<int32_t> result;
    inner_entity_->find_near_neighbors(converted_query, threshold, &result);
    return result;
  }

  std::vector<int32_t> get_unique_candidates(OuterNumPyArray q) {
    InnerVector converted_query = numpy_to_array_point(q);
    py::gil_scoped_release release;
    std::vector<int32_t> result;
    inner_entity_->get_unique_candidates(converted_query, &result);
    return result;
  }

  std::vector<int32_t> get_candidates_with_duplicates(OuterNumPyArray q) {
    InnerVector converted_query = numpy_to_array_point(q);
    py::gil_scoped_release release;
    std::vector<int32_t> result;
    inner_entity_->get_candidates_with_duplicates(converted_query, &result);
    return result;
  }

  void reset_query_statistics() {
    py::gil_scoped_release release;
    inner_entity_->reset_query_statistics();
  }

  QueryStatistics get_query_statistics() {
    py::gil_scoped_release release;
    return inner_entity_->get_query_statistics();
  }

 private:
  std::shared_ptr<InnerLSHQueryPool> inner_entity_;
};

typedef PyLSHNearestNeighborQueryDenseFloat OuterLSHQueryObject;
typedef PyLSHNearestNeighborQueryPoolDenseFloat OuterLSHQueryPool;

class PyLSHNearestNeighborTableDenseFloat {
 public:
  PyLSHNearestNeighborTableDenseFloat(std::shared_ptr<InnerLSHTable> table)
      : table_(table) {}

  std::unique_ptr<OuterLSHQueryObject> construct_query_object(
      int_fast64_t num_probes = -1,
      int_fast64_t max_num_candidates = -1) const {
    std::unique_ptr<InnerLSHQueryObject> inner_query_object =
        table_->construct_query_object(num_probes, max_num_candidates);
    return std::unique_ptr<OuterLSHQueryObject>(
        new OuterLSHQueryObject(std::move(inner_query_object)));
  }

  std::unique_ptr<OuterLSHQueryPool> construct_query_pool(
      int_fast64_t num_probes = -1, int_fast64_t max_num_candidates = -1,
      int_fast64_t num_query_objects = 0) const {
    std::unique_ptr<InnerLSHQueryPool> inner_query_pool =
        table_->construct_query_pool(num_probes, max_num_candidates,
                                     num_query_objects);
    return std::unique_ptr<OuterLSHQueryPool>(
        new OuterLSHQueryPool(std::move(inner_query_pool)));
  }

  void insert(OuterNumPyArray point) {
      InnerVector converted_point = numpy_to_array_point(point);
      table_->insert(converted_point);
  }

  void remove(int_fast64_t point_index) {
      table_->remove(point_index);
  }

 private:
  std::shared_ptr<InnerLSHTable> table_;
};

typedef PyLSHNearestNeighborTableDenseFloat OuterLSHTable;
InnerPointSet converted_setup_points;

std::unique_ptr<OuterLSHTable> construct_table_dense_float(
    OuterNumPyArray points, const LSHConstructionParameters &params) {
  converted_setup_points = numpy_to_array_dataset(points);
  std::unique_ptr<InnerLSHTable> inner_table =
      construct_table<InnerVector, int32_t, InnerPointSet>(
          converted_setup_points, params);
  return std::unique_ptr<OuterLSHTable>(
      new OuterLSHTable(std::move(inner_table)));
}

}  // namespace single_precision

namespace double_precision {

typedef double ScalarType;
typedef DenseVector<ScalarType> InnerVector;
//typedef EigenMap<ScalarType> InnerEigenMap;
//typedef PlainArrayPointSet<ScalarType> InnerPlainArrayPointSet;
typedef LSHTable<ScalarType> InnerLSHTable;
typedef LSHQueryObject<ScalarType> InnerLSHQueryObject;
typedef LSHQueryPool<ScalarType> InnerLSHQueryPool;
typedef NumPyArray<ScalarType> OuterNumPyArray;

typedef std::vector<InnerVector> InnerPointSet;
inline InnerPointSet numpy_to_array_dataset(OuterNumPyArray dataset) {
  py::buffer_info buf = dataset.request();
  if (buf.ndim != 2) {
    throw PyLSHNearestNeighborTableError("expected a two-dimensional array");
  }
  size_t num_points = buf.shape[0];
  size_t dimension = buf.shape[1];

  InnerPointSet converted_points;
  for (size_t i = 0; i < num_points; i++) {
      InnerVector v(dimension);
      for (size_t j = 0; j < dimension; j++) {
         v[j] = ((ScalarType *)buf.ptr)[i*dimension + j];
      }
      converted_points.push_back(v);
  }

//  free((ScalarType *)buf.ptr);
//  buf.ptr = NULL;

  return converted_points;
}

inline InnerVector numpy_to_array_point(OuterNumPyArray point) {
  py::buffer_info buf = point.request();
  if (buf.ndim != 1) {
    throw PyLSHNearestNeighborTableError("expected a two-dimensional array");
  }

  InnerVector converted_point(buf.shape[0]);
  for (long i = 0; i < buf.shape[0]; i++) {
     converted_point[i] = ((ScalarType *)buf.ptr)[i];
  }

  return converted_point;
}

class PyLSHNearestNeighborQueryDenseDouble {
 public:
  PyLSHNearestNeighborQueryDenseDouble(
      std::shared_ptr<InnerLSHQueryObject> query_object)
      : inner_entity_(query_object) {}

  void set_num_probes(int_fast64_t num_probes) {
    py::gil_scoped_release release;
    inner_entity_->set_num_probes(num_probes);
  }

  int_fast64_t get_num_probes() {
    py::gil_scoped_release release;
    return inner_entity_->get_num_probes();
  }

  void set_max_num_candidates(int_fast64_t max_num_candidates) {
    py::gil_scoped_release release;
    return inner_entity_->set_max_num_candidates(max_num_candidates);
  }

  int_fast64_t get_max_num_candidates() {
    py::gil_scoped_release release;
    return inner_entity_->get_max_num_candidates();
  }

  int32_t find_nearest_neighbor(OuterNumPyArray q) {
    InnerVector converted_query = numpy_to_array_point(q);
    py::gil_scoped_release release;
    return inner_entity_->find_nearest_neighbor(converted_query);
  }

  std::vector<std::pair<int32_t, float>> find_k_nearest_neighbors(OuterNumPyArray q,
                                                int_fast64_t k, float threshold) {
    InnerVector converted_query = numpy_to_array_point(q);
    py::gil_scoped_release release;
    std::vector<std::pair<int32_t, float>> result;
    inner_entity_->find_k_nearest_neighbors(converted_query, k, threshold, &result);
    return result;
  }

  std::vector<int32_t> find_near_neighbors(OuterNumPyArray q,
                                           ScalarType threshold) {
    InnerVector converted_query = numpy_to_array_point(q);
    py::gil_scoped_release release;
    std::vector<int32_t> result;
    inner_entity_->find_near_neighbors(converted_query, threshold, &result);
    return result;
  }

  std::vector<int32_t> get_unique_candidates(OuterNumPyArray q) {
    InnerVector converted_query = numpy_to_array_point(q);
    py::gil_scoped_release release;
    std::vector<int32_t> result;
    inner_entity_->get_unique_candidates(converted_query, &result);
    return result;
  }

  std::vector<int32_t> get_candidates_with_duplicates(OuterNumPyArray q) {
    InnerVector converted_query = numpy_to_array_point(q);
    py::gil_scoped_release release;
    std::vector<int32_t> result;
    inner_entity_->get_candidates_with_duplicates(converted_query, &result);
    return result;
  }

  void reset_query_statistics() {
    py::gil_scoped_release release;
    inner_entity_->reset_query_statistics();
  }

  QueryStatistics get_query_statistics() {
    py::gil_scoped_release release;
    return inner_entity_->get_query_statistics();
  }

 private:
  std::shared_ptr<InnerLSHQueryObject> inner_entity_;
};

class PyLSHNearestNeighborQueryPoolDenseDouble {
 public:
  PyLSHNearestNeighborQueryPoolDenseDouble(
      std::shared_ptr<InnerLSHQueryPool> query_pool)
      : inner_entity_(query_pool) {}

  void set_num_probes(int_fast64_t num_probes) {
    py::gil_scoped_release release;
    inner_entity_->set_num_probes(num_probes);
  }

  int_fast64_t get_num_probes() {
    py::gil_scoped_release release;
    return inner_entity_->get_num_probes();
  }

  void set_max_num_candidates(int_fast64_t max_num_candidates) {
    py::gil_scoped_release release;
    return inner_entity_->set_max_num_candidates(max_num_candidates);
  }

  int_fast64_t get_max_num_candidates() {
    py::gil_scoped_release release;
    return inner_entity_->get_max_num_candidates();
  }

  int32_t find_nearest_neighbor(OuterNumPyArray q) {
    InnerVector converted_query = numpy_to_array_point(q);
    py::gil_scoped_release release;
    return inner_entity_->find_nearest_neighbor(converted_query);
  }

  std::vector<std::pair<int32_t, float>> find_k_nearest_neighbors(OuterNumPyArray q,
                                                int_fast64_t k, float threshold) {
    InnerVector converted_query = numpy_to_array_point(q);
    py::gil_scoped_release release;
    std::vector<std::pair<int32_t, float>> result;
    inner_entity_->find_k_nearest_neighbors(converted_query, k, threshold, &result);
    return result;
  }

  std::vector<int32_t> find_near_neighbors(OuterNumPyArray q,
                                           ScalarType threshold) {
    InnerVector converted_query = numpy_to_array_point(q);
    py::gil_scoped_release release;
    std::vector<int32_t> result;
    inner_entity_->find_near_neighbors(converted_query, threshold, &result);
    return result;
  }

  std::vector<int32_t> get_unique_candidates(OuterNumPyArray q) {
    InnerVector converted_query = numpy_to_array_point(q);
    py::gil_scoped_release release;
    std::vector<int32_t> result;
    inner_entity_->get_unique_candidates(converted_query, &result);
    return result;
  }

  std::vector<int32_t> get_candidates_with_duplicates(OuterNumPyArray q) {
    InnerVector converted_query = numpy_to_array_point(q);
    py::gil_scoped_release release;
    std::vector<int32_t> result;
    inner_entity_->get_candidates_with_duplicates(converted_query, &result);
    return result;
  }

  void reset_query_statistics() {
    py::gil_scoped_release release;
    inner_entity_->reset_query_statistics();
  }

  QueryStatistics get_query_statistics() {
    py::gil_scoped_release release;
    return inner_entity_->get_query_statistics();
  }

 private:
  std::shared_ptr<InnerLSHQueryPool> inner_entity_;
};

typedef PyLSHNearestNeighborQueryDenseDouble OuterLSHQueryObject;
typedef PyLSHNearestNeighborQueryPoolDenseDouble OuterLSHQueryPool;

class PyLSHNearestNeighborTableDenseDouble {
 public:
  PyLSHNearestNeighborTableDenseDouble(std::shared_ptr<InnerLSHTable> table)
      : table_(table) {}

  std::unique_ptr<OuterLSHQueryObject> construct_query_object(
      int_fast64_t num_probes = -1,
      int_fast64_t max_num_candidates = -1) const {
    std::unique_ptr<InnerLSHQueryObject> inner_query_object =
        table_->construct_query_object(num_probes, max_num_candidates);
    return std::unique_ptr<OuterLSHQueryObject>(
        new OuterLSHQueryObject(std::move(inner_query_object)));
  }

  std::unique_ptr<OuterLSHQueryPool> construct_query_pool(
      int_fast64_t num_probes = -1, int_fast64_t max_num_candidates = -1,
      int_fast64_t num_query_objects = 0) const {
    std::unique_ptr<InnerLSHQueryPool> inner_query_pool =
        table_->construct_query_pool(num_probes, max_num_candidates,
                                     num_query_objects);
    return std::unique_ptr<OuterLSHQueryPool>(
        new OuterLSHQueryPool(std::move(inner_query_pool)));
  }

  void insert(OuterNumPyArray point) {
      InnerVector converted_point = numpy_to_array_point(point);
      table_->insert(converted_point);
  }

  void remove(int_fast64_t point_index) {
      table_->remove(point_index);
  }

 private:
  std::shared_ptr<InnerLSHTable> table_;
};

typedef PyLSHNearestNeighborTableDenseDouble OuterLSHTable;

std::unique_ptr<OuterLSHTable> construct_table_dense_double(
    OuterNumPyArray points, const LSHConstructionParameters &params) {
  InnerPointSet converted_points = numpy_to_array_dataset(points);
  std::unique_ptr<InnerLSHTable> inner_table =
      construct_table<InnerVector, int32_t, InnerPointSet>(
          converted_points, params);
  return std::unique_ptr<OuterLSHTable>(
      new OuterLSHTable(std::move(inner_table)));
}

}  // namespace double_precision

PYBIND11_MODULE(_falconn, m) {
  using single_precision::PyLSHNearestNeighborTableDenseFloat;
  using single_precision::PyLSHNearestNeighborQueryDenseFloat;
  using single_precision::PyLSHNearestNeighborQueryPoolDenseFloat;
  using single_precision::construct_table_dense_float;
  using double_precision::PyLSHNearestNeighborTableDenseDouble;
  using double_precision::PyLSHNearestNeighborQueryDenseDouble;
  using double_precision::PyLSHNearestNeighborQueryPoolDenseDouble;
  using double_precision::construct_table_dense_double;

  py::enum_<LSHFamily>(m, "LSHFamily")
      .value("Unknown", LSHFamily::Unknown)
      .value("Hyperplane", LSHFamily::Hyperplane)
      .value("CrossPolytope", LSHFamily::CrossPolytope);
  py::enum_<DistanceFunction>(m, "DistanceFunction")
      .value("Unknown", DistanceFunction::Unknown)
      .value("NegativeInnerProduct", DistanceFunction::NegativeInnerProduct)
      .value("EuclideanSquared", DistanceFunction::EuclideanSquared);
  py::enum_<StorageHashTable>(m, "StorageHashTable")
      .value("Unknown", StorageHashTable::Unknown)
      .value("FlatHashTable", StorageHashTable::FlatHashTable)
      .value("BitPackedFlatHashTable", StorageHashTable::BitPackedFlatHashTable)
      .value("STLHashTable", StorageHashTable::STLHashTable)
      .value("LinearProbingHashTable",
             StorageHashTable::LinearProbingHashTable);
  // we do not expose feature_hashing_dimension, since the wrapper does
  // not support sparse datasets yet
  py::class_<LSHConstructionParameters>(m, "LSHConstructionParameters")
      .def(py::init<>())
      .def_readwrite("dimension", &LSHConstructionParameters::dimension)
      .def_readwrite("lsh_family", &LSHConstructionParameters::lsh_family)
      .def_readwrite("distance_function",
                     &LSHConstructionParameters::distance_function)
      .def_readwrite("k", &LSHConstructionParameters::k)
      .def_readwrite("l", &LSHConstructionParameters::l)
      .def_readwrite("storage_hash_table",
                     &LSHConstructionParameters::storage_hash_table)
      .def_readwrite("num_setup_threads",
                     &LSHConstructionParameters::num_setup_threads)
      .def_readwrite("seed", &LSHConstructionParameters::seed)
      .def_readwrite("last_cp_dimension",
                     &LSHConstructionParameters::last_cp_dimension)
      .def_readwrite("num_rotations",
                     &LSHConstructionParameters::num_rotations);
  // we do not expose a constructor and make all the members read-only
  py::class_<QueryStatistics>(m, "QueryStatistics")
      .def_readonly("average_total_query_time",
                    &QueryStatistics::average_total_query_time)
      .def_readonly("average_lsh_time", &QueryStatistics::average_lsh_time)
      .def_readonly("average_hash_table_time",
                    &QueryStatistics::average_hash_table_time)
      .def_readonly("average_distance_time",
                    &QueryStatistics::average_distance_time)
      .def_readonly("average_num_candidates",
                    &QueryStatistics::average_num_candidates)
      .def_readonly("average_num_unique_candidates",
                    &QueryStatistics::average_num_unique_candidates)
      .def_readonly("num_queries", &QueryStatistics::num_queries);
  // we do not expose a constructor
  py::class_<PyLSHNearestNeighborTableDenseFloat>(
      m, "PyLSHNearestNeighborTableDenseFloat")
      .def("construct_query_object",
           &PyLSHNearestNeighborTableDenseFloat::construct_query_object,
           py::arg("num_probes") = -1, py::arg("max_num_candidates") = -1)
      .def("construct_query_pool",
           &PyLSHNearestNeighborTableDenseFloat::construct_query_pool,
           py::arg("num_probes") = -1, py::arg("max_num_candidates") = -1,
           py::arg("num_query_objects") = 0)
       .def("remove",
           &PyLSHNearestNeighborTableDenseFloat::remove,
           py::arg("point_index") = -1)
       .def("insert",
           &PyLSHNearestNeighborTableDenseFloat::insert,
           py::arg("point") = -1);
  // we do not expose a constructor
  py::class_<PyLSHNearestNeighborTableDenseDouble>(
      m, "PyLSHNearestNeighborTableDenseDouble")
      .def("construct_query_object",
           &PyLSHNearestNeighborTableDenseDouble::construct_query_object,
           py::arg("num_probes") = -1, py::arg("max_num_candidates") = -1)
      .def("construct_query_pool",
           &PyLSHNearestNeighborTableDenseDouble::construct_query_pool,
           py::arg("num_probes") = -1, py::arg("max_num_candidates") = -1,
           py::arg("num_query_objects") = 0)
      .def("remove",
           &PyLSHNearestNeighborTableDenseDouble::remove,
           py::arg("point_index") = -1)
       .def("insert",
           &PyLSHNearestNeighborTableDenseDouble::insert,
           py::arg("point") = -1);
  m.def("construct_table_dense_float", &construct_table_dense_float, "");
  m.def("construct_table_dense_double", &construct_table_dense_double, "");

  // we do not expose a constructor
  py::class_<PyLSHNearestNeighborQueryDenseFloat>(
      m, "PyLSHNearestNeighborQueryDenseFloat")
      .def("set_num_probes",
           &PyLSHNearestNeighborQueryDenseFloat::set_num_probes)
      .def("get_num_probes",
           &PyLSHNearestNeighborQueryDenseFloat::get_num_probes)
      .def("set_max_num_candidates",
           &PyLSHNearestNeighborQueryDenseFloat::set_max_num_candidates)
      .def("get_max_num_candidates",
           &PyLSHNearestNeighborQueryDenseFloat::get_max_num_candidates)
      .def("find_nearest_neighbor",
           &PyLSHNearestNeighborQueryDenseFloat::find_nearest_neighbor)
      .def("find_k_nearest_neighbors",
           &PyLSHNearestNeighborQueryDenseFloat::find_k_nearest_neighbors)
      .def("find_near_neighbors",
           &PyLSHNearestNeighborQueryDenseFloat::find_near_neighbors)
      .def("get_unique_candidates",
           &PyLSHNearestNeighborQueryDenseFloat::get_unique_candidates)
      .def("get_candidates_with_duplicates",
           &PyLSHNearestNeighborQueryDenseFloat::get_candidates_with_duplicates)
      .def("reset_query_statistics",
           &PyLSHNearestNeighborQueryDenseFloat::reset_query_statistics)
      .def("get_query_statistics",
           &PyLSHNearestNeighborQueryDenseFloat::get_query_statistics);

  // we do not expose a constructor
  py::class_<PyLSHNearestNeighborQueryPoolDenseFloat>(
      m, "PyLSHNearestNeighborQueryPoolDenseFloat")
      .def("set_num_probes",
           &PyLSHNearestNeighborQueryPoolDenseFloat::set_num_probes)
      .def("get_num_probes",
           &PyLSHNearestNeighborQueryPoolDenseFloat::get_num_probes)
      .def("set_max_num_candidates",
           &PyLSHNearestNeighborQueryPoolDenseFloat::set_max_num_candidates)
      .def("get_max_num_candidates",
           &PyLSHNearestNeighborQueryPoolDenseFloat::get_max_num_candidates)
      .def("find_nearest_neighbor",
           &PyLSHNearestNeighborQueryPoolDenseFloat::find_nearest_neighbor)
      .def("find_k_nearest_neighbors",
           &PyLSHNearestNeighborQueryPoolDenseFloat::find_k_nearest_neighbors)
      .def("find_near_neighbors",
           &PyLSHNearestNeighborQueryPoolDenseFloat::find_near_neighbors)
      .def("get_unique_candidates",
           &PyLSHNearestNeighborQueryPoolDenseFloat::get_unique_candidates)
      .def("get_candidates_with_duplicates",
           &PyLSHNearestNeighborQueryPoolDenseFloat::
               get_candidates_with_duplicates)
      .def("reset_query_statistics",
           &PyLSHNearestNeighborQueryPoolDenseFloat::reset_query_statistics)
      .def("get_query_statistics",
           &PyLSHNearestNeighborQueryPoolDenseFloat::get_query_statistics);

  // we do not expose a constructor
  py::class_<PyLSHNearestNeighborQueryDenseDouble>(
      m, "PyLSHNearestNeighborQueryDenseDouble")
      .def("set_num_probes",
           &PyLSHNearestNeighborQueryDenseDouble::set_num_probes)
      .def("get_num_probes",
           &PyLSHNearestNeighborQueryDenseDouble::get_num_probes)
      .def("set_max_num_candidates",
           &PyLSHNearestNeighborQueryDenseDouble::set_max_num_candidates)
      .def("get_max_num_candidates",
           &PyLSHNearestNeighborQueryDenseDouble::get_max_num_candidates)
      .def("find_nearest_neighbor",
           &PyLSHNearestNeighborQueryDenseDouble::find_nearest_neighbor)
      .def("find_k_nearest_neighbors",
           &PyLSHNearestNeighborQueryDenseDouble::find_k_nearest_neighbors)
      .def("find_near_neighbors",
           &PyLSHNearestNeighborQueryDenseDouble::find_near_neighbors)
      .def("get_unique_candidates",
           &PyLSHNearestNeighborQueryDenseDouble::get_unique_candidates)
      .def(
          "get_candidates_with_duplicates",
          &PyLSHNearestNeighborQueryDenseDouble::get_candidates_with_duplicates)
      .def("reset_query_statistics",
           &PyLSHNearestNeighborQueryDenseDouble::reset_query_statistics)
      .def("get_query_statistics",
           &PyLSHNearestNeighborQueryDenseDouble::get_query_statistics);

  // we do not expose a constructor
  py::class_<PyLSHNearestNeighborQueryPoolDenseDouble>(
      m, "PyLSHNearestNeighborQueryPoolDenseDouble")
      .def("set_num_probes",
           &PyLSHNearestNeighborQueryPoolDenseDouble::set_num_probes)
      .def("get_num_probes",
           &PyLSHNearestNeighborQueryPoolDenseDouble::get_num_probes)
      .def("set_max_num_candidates",
           &PyLSHNearestNeighborQueryPoolDenseDouble::set_max_num_candidates)
      .def("get_max_num_candidates",
           &PyLSHNearestNeighborQueryPoolDenseDouble::get_max_num_candidates)
      .def("find_nearest_neighbor",
           &PyLSHNearestNeighborQueryPoolDenseDouble::find_nearest_neighbor)
      .def("find_k_nearest_neighbors",
           &PyLSHNearestNeighborQueryPoolDenseDouble::find_k_nearest_neighbors)
      .def("find_near_neighbors",
           &PyLSHNearestNeighborQueryPoolDenseDouble::find_near_neighbors)
      .def("get_unique_candidates",
           &PyLSHNearestNeighborQueryPoolDenseDouble::get_unique_candidates)
      .def("get_candidates_with_duplicates",
           &PyLSHNearestNeighborQueryPoolDenseDouble::
               get_candidates_with_duplicates)
      .def("reset_query_statistics",
           &PyLSHNearestNeighborQueryPoolDenseDouble::reset_query_statistics)
      .def("get_query_statistics",
           &PyLSHNearestNeighborQueryPoolDenseDouble::get_query_statistics);

  m.def("compute_number_of_hash_functions",
        &compute_number_of_hash_functions<DenseVector<float>>, "",
        py::arg("num_hash_bits"), py::arg("params"));
  m.def("get_default_parameters", &get_default_parameters<DenseVector<float>>,
        "", py::arg("num_points"), py::arg("dimension"),
        py::arg("distance") = DistanceFunction::EuclideanSquared,
        py::arg("is_sufficiently_dense") = false);
}
}  // namespace python
}  // namespace falconn
