import falconn
import numpy as np

def test_number_of_hash_functions():
  params = falconn._internal.LSHConstructionParameters()
  
  params.lsh_family = 'hyperplane'
  params.dimension = 10
  falconn.compute_number_of_hash_functions(5, params)
  assert params.k == 5
  
  params.lsh_family = 'cross_polytope'
  falconn.compute_number_of_hash_functions(5, params)
  assert params.k == 1
  assert params.last_cp_dimension == 16

  params.dimension = 100
  params.lsh_family = 'hyperplane'
  falconn.compute_number_of_hash_functions(8, params)
  assert params.k == 8
  
  params.lsh_family = 'cross_polytope'
  falconn.compute_number_of_hash_functions(8, params)
  assert params.k == 1
  assert params.last_cp_dimension == 128

  falconn.compute_number_of_hash_functions(10, params)
  assert params.k == 2
  assert params.last_cp_dimension == 2


def test_get_default_parameters():
  n = 100000
  dim = 128
  dist_func = 'negative_inner_product'
  params = falconn.get_default_parameters(n, dim, dist_func, True)
  assert params.l == 10
  assert params.lsh_family == 'cross_polytope'
  assert params.k == 2
  assert params.dimension == dim
  assert params.distance_function == dist_func
  assert params.num_rotations == 1
  assert params.last_cp_dimension == 64

def test_lsh_index_positive():
  n = 1000
  d = 128
  p = falconn.get_default_parameters(n, d)
  t = falconn.LSHIndex(p)
  dataset = np.random.randn(n, d).astype(np.float32)
  t.fit(dataset)
  u = np.random.randn(d).astype(np.float32)
  t.find_k_nearest_neighbors(u, 10)
  t.find_near_neighbors(u, 10.0)
  t.find_nearest_neighbor(u)
  t.get_candidates_with_duplicates(u)
  t.get_max_num_candidates()
  t.get_num_probes()
  t.get_query_statistics()
  t.get_unique_candidates(u)
  t.get_unique_sorted_candidates(u)
  t.reset_query_statistics()
  t.set_max_num_candidates(100)
  t.set_num_probes(10)

def test_lsh_index_negative():
  n = 1000
  d = 128
  p = falconn.get_default_parameters(n, d)
  t = falconn.LSHIndex(p)
  try:
    t.find_nearest_neighbor(np.random.randn(d))
    assert False
  except RuntimeError:
    pass
  try:
    dataset = [[1.0, 2.0], [3.0, 4.0]]
    t.fit(dataset)
    assert False
  except TypeError:
    pass
  try:
    dataset = np.random.randn(n, d).astype(np.int32)
    t.fit(dataset)
    assert False
  except ValueError:
    pass
  try:
    dataset = np.random.randn(10, 10, 10)
    t.fit(dataset)
    assert False
  except ValueError:
    pass
  dataset = np.random.randn(n, d).astype(np.float32)
  t.fit(dataset)
  dataset = np.random.randn(n, d).astype(np.float64)
  t.fit(dataset)
  u = np.random.randn(d).astype(np.float64)
  
  try:
    t.find_k_nearest_neighbors(u, 0.5)
    assert False
  except TypeError:
    pass

  try:
    t.find_k_nearest_neighbors(u, -1)
    assert False
  except ValueError:
    pass
  
  try:
    t.find_near_neighbors(u, -1)
    assert False
  except ValueError:
    pass
  
  try:
    t.set_max_num_candidates(0.5)
    assert False
  except TypeError:
    pass
  try:
    t.set_max_num_candidates(-10)
    assert False
  except ValueError:
    pass
  t.set_num_probes(t._params.l)
  try:
    t.set_num_probes(t._params.l - 1)
    assert False
  except ValueError:
    pass

  def check_check_query(f):
    try:
      f(u.astype(np.float32))
      assert False
    except ValueError:
      pass
    try:
      f([0.0] * d)
      assert False
    except TypeError:
      pass
    try:
      f(u[:d-1])
      assert False
    except ValueError:
      pass
    try:
      f(np.random.randn(d, d))
      assert False
    except ValueError:
      pass

  check_check_query(lambda u: t.find_k_nearest_neighbors(u, 10))
  check_check_query(lambda u: t.find_near_neighbors(u, 0.5))
  check_check_query(lambda u: t.find_nearest_neighbor(u))
  check_check_query(lambda u: t.get_candidates_with_duplicates(u))
  check_check_query(lambda u: t.get_unique_candidates(u))
  check_check_query(lambda u: t.get_unique_sorted_candidates(u))
  t.find_near_neighbors(u, 0.0)
