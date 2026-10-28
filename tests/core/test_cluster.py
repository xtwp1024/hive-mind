import time
import pytest
from hive.core.cluster import Cluster, ClusterManager, ClusterType

def test_cluster_basic_operations():
    cluster = Cluster(name="TestCluster", type=ClusterType.TASK_ORIENTED)
    assert cluster.is_active is True
    assert cluster.size == 0
    
    cluster.add_member("atom-1")
    assert cluster.size == 1
    assert "atom-1" in cluster.member_ids
    
    cluster.update_task("Coding", 0.5)
    assert cluster.current_task == "Coding"
    assert cluster.task_progress == 0.5
    
    cluster.complete_task()
    assert cluster.current_task == ""
    assert cluster.tasks_completed == 1
    
    cluster.dissolve()
    assert cluster.is_active is False
    assert cluster.size == 0

def test_cluster_serialization():
    cluster = Cluster(name="SerialTest")
    cluster.add_member("bee-123")
    
    d = cluster.to_dict()
    assert d["name"] == "SerialTest"
    assert "bee-123" in d["member_ids"]
    
    new_cluster = Cluster.from_dict(d)
    assert new_cluster.id == cluster.id
    assert "bee-123" in new_cluster.member_ids

def test_cluster_manager_lifecycle():
    manager = ClusterManager()
    manager.CLUSTER_TIMEOUT = 1 # 缩短超时用于测试
    
    c = manager.create_cluster(["a1", "a2"], ClusterType.TEMPORARY)
    assert c.id in manager.clusters
    assert manager.get_cluster(c.id) == c
    
    # 测试超时清理
    time.sleep(1.1)
    manager._cleanup_inactive(0)
    assert c.id not in manager.clusters

def test_cluster_manager_max_limit():
    manager = ClusterManager()
    manager.MAX_CLUSTERS = 2
    
    c1 = manager.create_cluster(["m1"])
    time.sleep(0.01)
    c2 = manager.create_cluster(["m2"])
    time.sleep(0.01)
    c3 = manager.create_cluster(["m3"])
    
    # 应该移除了最老的 c1
    assert c1.id not in manager.clusters
    assert c2.id in manager.clusters
    assert c3.id in manager.clusters

def test_cluster_overlap_detection():
    manager = ClusterManager()
    c1 = manager.create_cluster(["p1", "p2", "p3"])
    
    # 更新列表，其中 2/3 成员重合，应识别为同一簇
    new_lists = [["p1", "p2", "p4"]]
    manager.update_clusters(new_lists, 1)
    
    active = manager.get_active_clusters()
    assert len(active) == 1
    assert active[0].id == c1.id
    assert "p4" in active[0].member_ids
