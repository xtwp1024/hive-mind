# -*- coding: utf-8 -*-
"""P0-3: offline execution of the real sync assignment / async cycle boundary.

Run directly with stdlib unittest, or collect with pytest hive/tests/.
Production functions are compiled from AST without importing the application:
no credentials, network, worker processes, runtime state or third-party packages.
Only external I/O is replaced; hunt_sync still executes the real asyncio.run.
"""

import __future__

import ast
import asyncio
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]


def load_function(path, name, namespace, owner=None, old_boundary=False):
    """Extract the actual function body, including its runtime imports."""
    tree = ast.parse((ROOT / path).read_bytes(), filename=path)
    body = tree.body
    if owner:
        cls = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == owner)
        body = cls.body
    node = next(
        node
        for node in body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    if old_boundary:
        assignments = [
            item
            for item in node.body
            if isinstance(item, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "dna_batch" for t in item.targets)
        ]
        if len(assignments) != 1:
            raise AssertionError("cycle assignment boundary changed; update the counterexample")
        assignments[0].value = ast.parse(
            "_handle_warmup_phase(cycle_count, engineer)", mode="eval"
        ).body
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, path, "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


class AssignmentHarness:
    """Keep the real orchestration; replace disk/network/deployment boundaries."""

    def __init__(self, idle_count=40, prey=None, hunt_error=None):
        self.logger = Mock()
        self.trace = []
        self.deployed = []
        self.assignment_threads = []
        self.hunt_loops = []
        self.coroutines = []
        self.idle = [{"idle": index} for index in range(idle_count)]
        self.prey = [{"task": "offline-real-task"}] if prey is None else prey
        self.hunt_error = hunt_error
        self.casualties = []
        self.backlog = []
        self.started = None
        self.release = None
        harness = self

        async def offline_hunt(keywords, count):
            self.hunt_loops.append(asyncio.get_running_loop())
            self.trace.append(("hunt", count))
            await asyncio.sleep(0)
            if self.hunt_error:
                raise self.hunt_error
            return self.prey

        def hunt(_hunter, keywords, count):
            coroutine = offline_hunt(keywords, count)
            self.coroutines.append(coroutine)
            return coroutine

        hunter_namespace = {"asyncio": asyncio}
        hunt_sync = load_function(
            "hive/utils/github_hunter.py", "hunt_sync", hunter_namespace, "GithubHunter"
        )
        self.hunter_class = type("OfflineHunter", (), {"hunt": hunt, "hunt_sync": hunt_sync})
        self.engineer = types.SimpleNamespace(
            _check_morgue=self.check_morgue,
            auto_feast=types.SimpleNamespace(feast=self.feast),
            _roll_diverse_mutation=lambda: "standard",
            _determine_hunting_topics=lambda topics: topics or ["offline"],
            _filter_unique_prey=lambda prey, limit: prey[:limit],
            _update_intel_strategy=lambda prey, topics: None,
            _create_hunter_tasks=lambda batch, prey: batch.extend(prey),
        )
        engineer_namespace = {"logger": self.logger}
        for name in ("assign_tasks", "_process_backlog", "_perform_active_hunting"):
            function = load_function(
                "hive/ministers/engineer.py", name, engineer_namespace, "EngineerBee"
            )
            setattr(self.engineer, name, types.MethodType(function, self.engineer))

        self.warmup = types.SimpleNamespace(
            generate_full_swarm=self.generate_idle,
            get_species_status=lambda: {"total_species": idle_count},
        )
        warmup_function = load_function(
            "hive/core/warmup.py", "create_active_swarm", {"logger": self.logger}, "HiveWarmup"
        )
        self.warmup.create_active_swarm = types.MethodType(warmup_function, self.warmup)

        async def no_rest(_seconds):
            await asyncio.sleep(0)

        self.namespace = {
            "asyncio": types.SimpleNamespace(
                get_running_loop=asyncio.get_running_loop, sleep=no_rest
            ),
            "logger": self.logger,
            "get_warmup": lambda: self.warmup,
            "emit": lambda name, **data: self.trace.append((name, data.get("cycle"))),
            "_handle_deployment_phase": self.deploy,
            "_update_metrics_and_health": lambda *args: self.trace.append(("metrics", None)),
        }
        load_function("main.py", "_handle_warmup_phase", self.namespace)
        self.cycle = load_function("main.py", "_execute_single_cycle", self.namespace)
        self.queen = types.SimpleNamespace(get_colony_status=lambda: {})
        exports = {
            "hive.core.service_locator": {
                "BeeServiceLocator": types.SimpleNamespace(get_service=lambda name: None)
            },
            "hive.utils.github_hunter": {"GithubHunter": self.hunter_class},
            "hive.ministers.engineer": {"get_engineer": lambda: harness.engineer},
        }
        self.modules = {}
        for name, attributes in exports.items():
            module = types.ModuleType(name)
            module.__dict__.update(attributes)
            self.modules[name] = module

    def check_morgue(self, limit):
        self.assignment_threads.append(threading.get_ident())
        self.trace.append(("casualties", limit))
        if self.started is not None:
            self.started.set()
            if not self.release.wait(timeout=3):
                raise TimeoutError("test did not release assignment thread")
        return self.casualties[:limit]

    def feast(self, limit):
        self.trace.append(("backlog", limit))
        return self.backlog[:limit]

    def generate_idle(self):
        self.trace.append(("warmup", len(self.idle)))
        return self.idle.copy()

    async def deploy(self, queen, batch):
        self.trace.append(("deploy", len(batch)))
        self.deployed.append(batch)

    async def run_cycles(self, *cycles, old_boundary=False):
        function = self.cycle
        if old_boundary:
            function = load_function(
                "main.py", "_execute_single_cycle", self.namespace, old_boundary=True
            )
        try:
            with patch.dict(sys.modules, self.modules):
                for cycle in cycles:
                    await function(cycle, max(cycles), self.queen, self.engineer, None, None)
        finally:
            # The negative control creates a coroutine before asyncio.run rejects
            # the nested loop. Close it explicitly, rather than suppress warnings.
            for coroutine in self.coroutines:
                coroutine.close()


class TestMainTaskAssignmentAsync(unittest.IsolatedAsyncioTestCase):
    async def test_second_cycle_runs_real_sync_bridge_outside_main_loop(self):
        harness = AssignmentHarness()
        main_thread = threading.get_ident()
        main_loop = asyncio.get_running_loop()
        await harness.run_cycles(2)
        self.assertEqual(harness.deployed, [harness.prey])
        self.assertIs(harness.deployed[0][0], harness.prey[0])
        self.assertTrue(harness.assignment_threads)
        self.assertTrue(all(thread != main_thread for thread in harness.assignment_threads))
        self.assertEqual(len(harness.hunt_loops), 1)
        self.assertIsNot(harness.hunt_loops[0], main_loop)
        self.assertEqual(
            [
                name
                for name, _ in harness.trace
                if name in ("casualties", "backlog", "hunt", "deploy")
            ],
            ["casualties", "backlog", "hunt", "deploy"],
        )
        harness.logger.error.assert_not_called()

    async def test_two_cycles_complete_in_order_without_overlap(self):
        harness = AssignmentHarness()
        await harness.run_cycles(1, 2)
        events = [(name, value) for name, value in harness.trace if name.startswith("cycle_")]
        self.assertEqual(
            events,
            [("cycle_start", 1), ("cycle_end", 1), ("cycle_start", 2), ("cycle_end", 2)],
        )
        self.assertEqual(harness.deployed, [harness.idle, harness.prey])
        self.assertEqual(len(harness.hunt_loops), 1)
        harness.logger.error.assert_not_called()

    async def test_first_cycle_warmup_remaining_capacity_can_hunt(self):
        harness = AssignmentHarness(idle_count=29)
        await harness.run_cycles(1)
        self.assertEqual(harness.deployed, [harness.idle + harness.prey])
        self.assertIn(("hunt", 2), harness.trace)
        self.assertTrue(
            all(thread != threading.get_ident() for thread in harness.assignment_threads)
        )
        harness.logger.error.assert_not_called()

    async def test_empty_hunt_keeps_idle_fallback(self):
        harness = AssignmentHarness(prey=[])
        await harness.run_cycles(2)
        self.assertEqual(harness.deployed, [harness.idle])
        self.assertEqual(len(harness.hunt_loops), 1)
        harness.logger.error.assert_not_called()

    async def test_assignment_error_keeps_idle_fallback(self):
        harness = AssignmentHarness(hunt_error=ValueError("offline search failure"))
        await harness.run_cycles(2)
        self.assertEqual(harness.deployed, [harness.idle])
        harness.logger.error.assert_called_once()
        self.assertIn("offline search failure", harness.logger.error.call_args.args[0])

    async def test_event_loop_responds_while_assignment_waits(self):
        harness = AssignmentHarness()
        harness.started = threading.Event()
        harness.release = threading.Event()
        task = asyncio.create_task(harness.run_cycles(2))

        async def wait_for_assignment():
            while not harness.started.is_set():
                await asyncio.sleep(0)

        try:
            await asyncio.wait_for(wait_for_assignment(), timeout=2)
            self.assertFalse(task.done())
            self.assertEqual(harness.deployed, [])
            # Reaching here before release proves the main loop is not blocked.
            self.assertFalse(harness.release.is_set())
        finally:
            harness.release.set()
            await asyncio.wait_for(task, timeout=4)
        self.assertEqual(harness.deployed, [harness.prey])
        harness.logger.error.assert_not_called()

    async def test_old_boundary_reproduces_nested_loop_and_false_idle_success(self):
        harness = AssignmentHarness()
        await harness.run_cycles(2, old_boundary=True)
        self.assertEqual(harness.deployed, [harness.idle])
        self.assertEqual(harness.hunt_loops, [])
        harness.logger.error.assert_called_once()
        self.assertIn(
            "asyncio.run() cannot be called from a running event loop",
            harness.logger.error.call_args.args[0],
        )

    async def test_repairs_take_priority_and_reduce_remaining_capacity(self):
        harness = AssignmentHarness()
        harness.casualties = [{"repair": index} for index in range(15)]
        harness.backlog = [types.SimpleNamespace(mutation_gene=None) for _ in range(14)]
        await harness.run_cycles(2)
        self.assertEqual(harness.deployed, [harness.casualties + harness.backlog + harness.prey])
        self.assertIn(("backlog", 15), harness.trace)
        self.assertIn(("hunt", 2), harness.trace)
        harness.logger.error.assert_not_called()

    async def test_backlog_fills_capacity_before_hunting(self):
        harness = AssignmentHarness()
        harness.backlog = [types.SimpleNamespace(mutation_gene=None) for _ in range(30)]
        await harness.run_cycles(2)
        self.assertEqual(harness.deployed, [harness.backlog])
        self.assertTrue(all(dna.mutation_gene == "standard" for dna in harness.backlog))
        self.assertEqual(harness.hunt_loops, [])
        harness.logger.error.assert_not_called()


class TestSyncAssignmentCompatibility(unittest.TestCase):
    def test_existing_sync_assignment_still_works_without_running_loop(self):
        harness = AssignmentHarness()
        with patch.dict(sys.modules, harness.modules):
            batch = harness.engineer.assign_tasks(capacity=30)
        self.assertEqual(batch, harness.prey)
        self.assertEqual(harness.assignment_threads, [threading.get_ident()])
        self.assertEqual(len(harness.hunt_loops), 1)
        harness.logger.error.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
