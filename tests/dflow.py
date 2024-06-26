from dataclasses import dataclass
from typing import Annotated, Optional
import unittest

from dflow_galaxy.core import dflow_builder, types


class TestDdflow(unittest.TestCase):

    def test_pickle_form(self):
        import cloudpickle as cp
        import bz2
        import base64

        @dataclass
        class Foo:
            x: int

        def foo(x):
            return x

        Foo_pkl = dflow_builder.pickle_converts(Foo)
        r = eval(Foo_pkl)(1)
        self.assertIsInstance(r, Foo)
        self.assertEqual(r.x, 1)

        foo_pkl = dflow_builder.pickle_converts(foo)
        r = eval(foo_pkl)(1)
        self.assertEqual(r, 1)

    def test_valid_python_step_input(self):
        @dataclass(frozen=True)
        class Foo:
            x: types.InputParam[int]
            y: types.InputArtifact
            z: types.OutputArtifact
        foo = Foo(1, '2', '3')
        list(dflow_builder.iter_python_step_args(foo))

    def test_invalid_python_step_input(self):
        @dataclass(frozen=True)
        class Foo:
            x: types.InputParam[int]
            y: types.InputArtifact
            z: types.OutputArtifact
            e: int

        @dataclass(frozen=True)
        class Bar:
            x: types.InputParam[int]
            y: types.InputArtifact
            z: types.OutputArtifact
            e: types.OutputParam[int]

        with self.assertRaises(ValueError):
            list(dflow_builder.iter_python_step_args(Foo(1, '2', '3', 4)))

        with self.assertRaises(ValueError):
            list(dflow_builder.iter_python_step_args(Bar(1, '2', '3', 4)))

    def test_valid_python_step_output(self):
        @dataclass
        class Foo:
            x: types.OutputParam[int]
        foo = Foo(1)
        list(dflow_builder.iter_python_step_return(foo))

    def test_invalid_python_step_output(self):
        @dataclass
        class Foo:
            x: types.OutputParam[int]
            y: int
        with self.assertRaises(AssertionError):
            list(dflow_builder.iter_python_step_return(Foo(1, 2)))

    def test_convert_to_argo_script(self):
        import dflow

        @dataclass(frozen=True)
        class FooInput:
            x: types.InputParam[int]
            y: types.InputArtifact
            z: types.OutputArtifact
            u: Annotated[str, dflow.InputArtifact()]
            v: Optional[types.InputArtifact]

        @dataclass
        class FooOutput:
            x: types.OutputParam[int]

        def foo(input: FooInput) -> FooOutput:
            return FooOutput(input.x)

        ret = dflow_builder.python_build_template(foo, base_dir='/tmp/dflow-galaxy')
        print(ret.source)

    def test_argo_script_without_return(self):
        @dataclass(frozen=True)
        class FooInput:
            ...
        def foo(input: FooInput):
            pass
        def foo2(input: FooInput) -> None:
            return None

        dflow_builder.python_build_template(foo, base_dir='/tmp/dflow-galaxy')
        dflow_builder.python_build_template(foo2, base_dir='/tmp/dflow-galaxy')

    def test_bash_build_template(self):
        import dflow

        @dataclass(frozen=True)
        class FooArgs:
            x: types.InputParam[int]
            y: types.InputArtifact
            z: types.OutputArtifact
            u: Annotated[str, dflow.InputArtifact()]
            v: Optional[types.InputArtifact]

        def foo(args: FooArgs) -> str:
            return f'''\
echo "{args.x}"
echo "{args.y}"
echo "{args.z}"
echo "{args.u}"
echo "{args.v}"
'''
        ret = dflow_builder.bash_build_template(foo, base_dir='/tmp/dflow-galaxy')
        print(ret.source)


if __name__ == '__main__':
    unittest.main()
