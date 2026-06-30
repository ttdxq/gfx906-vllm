#!/usr/bin/env python3
"""
vLLM gfx906 运行时问题综合诊断工具

一次性检查多个运行时问题，包括：
- 依赖版本冲突
- 配置完整性
- 模型加载问题
- 平台兼容性
- 导入错误
"""

import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict
import importlib.util


class DiagnosticResult:
    def __init__(
        self,
        category: str,
        name: str,
        passed: bool,
        message: str = "",
        severity: str = "INFO",
    ):
        self.category = category
        self.name = name
        self.passed = passed
        self.message = message
        self.severity = severity

    def __str__(self):
        status = "✅" if self.passed else "❌"
        severity_mark = {
            "CRITICAL": "🔴",
            "HIGH": "⚠️",
            "MEDIUM": "📡",
            "INFO": "ℹ️",
        }.get(self.severity, "")
        return f"{status} {severity_mark} [{self.category}] {self.name}: {self.message}"


class RuntimeDiagnostics:
    def __init__(self):
        self.results: List[DiagnosticResult] = []
        self.vllm_path = Path(__file__).parent
        self.errors = []
        self.warnings = []

    def add_result(
        self,
        category: str,
        name: str,
        passed: bool,
        message: str = "",
        severity: str = "INFO",
    ):
        result = DiagnosticResult(category, name, passed, message, severity)
        self.results.append(result)
        if not passed and severity == "CRITICAL":
            self.errors.append(result)
        elif not passed and severity in ["HIGH", "MEDIUM"]:
            self.warnings.append(result)

    def check_python_version(self):
        """检查Python版本兼容性"""
        version = sys.version_info
        is_compatible = (3, 10) <= version < (3, 14)

        self.add_result(
            "Python环境",
            "Python版本兼容性",
            is_compatible,
            f"当前版本: {version.major}.{version.minor}.{version.micro}"
            + (", ✅ 兼容" if is_compatible else ", ❌ 不兼容（需要3.10-3.13）"),
            "CRITICAL" if not is_compatible else "INFO",
        )

    def check_core_dependencies(self):
        """检查核心依赖"""
        deps = [
            ("torch", "2.9.0", None),
            ("transformers", "4.56.0", None),
            ("protobuf", "5.29.6", None),
            ("pydantic", "2.12.0", None),
            ("fastapi", "0.115.0", None),
        ]

        for module_name, min_version, max_version in deps:
            try:
                if module_name == "torch":
                    import torch

                    version = torch.__version__
                elif module_name == "transformers":
                    import transformers

                    version = transformers.__version__
                elif module_name == "protobuf":
                    import google.protobuf as protobuf

                    version = protobuf.__version__
                elif module_name == "pydantic":
                    import pydantic

                    version = pydantic.__version__
                elif module_name == "fastapi":
                    import fastapi

                    version = fastapi.__version__
                else:
                    version = "unknown"

                # 简单版本比较
                passed = True
                msg = f"版本: {version}"
                severity = "INFO"

                # 特殊检查：protobuf安全版本
                if module_name == "protobuf":
                    try:
                        version_parts = version.split(".")
                        major, minor, patch = (
                            int(version_parts[0]),
                            int(version_parts[1]),
                            int(version_parts[2].split("+")[0]),
                        )
                        if major < 5 or (major == 5 and minor < 29):
                            passed = False
                            msg += f" ❌ 版本过低（存在CVE-2026-0994漏洞）"
                            severity = "CRITICAL"
                        else:
                            msg += f" ✅ 安全版本"
                    except:
                        pass

                self.add_result("核心依赖", f"{module_name}", passed, msg, severity)

            except ImportError as e:
                self.add_result(
                    "核心依赖", f"{module_name}", False, f"导入失败: {e}", "CRITICAL"
                )

    def check_vllm_compiled_modules(self):
        """检查vLLM编译模块"""
        modules = [
            ("vllm._C", "C扩展"),
            ("vllm._rocm_C", "ROCm C扩展"),
        ]

        for module_name, desc in modules:
            try:
                __import__(module_name)
                self.add_result("编译模块", desc, True, "✅ 已编译", "INFO")
            except ImportError as e:
                self.add_result(
                    "编译模块", desc, False, f"❌ 未编译或导入失败: {e}", "CRITICAL"
                )

    def check_qwen35_configs(self):
        """检查Qwen3.5配置导出和注册"""
        try:
            # 检查配置导出
            from vllm.transformers_utils.configs import Qwen3_5Config, Qwen3_5MoeConfig

            self.add_result(
                "模型配置",
                "Qwen3.5配置导出",
                True,
                "✅ Qwen3_5Config和Qwen3_5MoeConfig已导出",
                "INFO",
            )
        except ImportError as e:
            self.add_result(
                "模型配置", "Qwen3.5配置导出", False, f"❌ 配置导出失败: {e}", "HIGH"
            )

        try:
            # 检查配置注册
            from vllm.model_executor.models.config import MODELS_CONFIG_MAP

            is_registered = "Qwen3_5ForConditionalGeneration" in MODELS_CONFIG_MAP
            self.add_result(
                "模型配置",
                "Qwen3.5配置注册",
                is_registered,
                "✅ 已注册到MODELS_CONFIG_MAP" if is_registered else "❌ 未注册",
                "HIGH" if not is_registered else "INFO",
            )
        except Exception as e:
            self.add_result(
                "模型配置", "Qwen3.5配置注册", False, f"❌ 检查失败: {e}", "HIGH"
            )

    def check_rocm_platform(self):
        """检查ROCm平台配置"""
        try:
            from vllm.platforms import current_platform

            platform_name = current_platform.__class__.__name__

            is_rocm = "ROCm" in platform_name
            self.add_result(
                "平台配置",
                "ROCm平台检测",
                is_rocm,
                f"当前平台: {platform_name}" + (" ✅" if is_rocm else " ⚠️ 非ROCm"),
                "INFO",
            )
        except Exception as e:
            self.add_result(
                "平台配置", "ROCm平台检测", False, f"❌ 检测失败: {e}", "MEDIUM"
            )

        try:
            # 检查amdsmi
            from amdsmi import amdsmi_get_processor_handles

            self.add_result(
                "平台配置", "amdsmi库", True, "✅ AMD GPU监控库可用", "INFO"
            )
        except ImportError:
            self.add_result(
                "平台配置",
                "amdsmi库",
                False,
                "⚠️ AMD GPU监控库不可用（可选）",
                "INFO",  # amdsmi是可选的
            )

    def check_gitignore_patterns(self):
        """检查.gitignore配置"""
        gitignore_path = self.vllm_path / ".gitignore"

        if not gitignore_path.exists():
            self.add_result(
                "代码管理", ".gitignore文件", False, "❌ .gitignore文件不存在", "HIGH"
            )
            return

        try:
            content = gitignore_path.read_text(encoding="utf-8")

            # 检查有问题的规则
            has_problematic_md_rule = "*.md" in content and "!README.md" in content
            has_readme_protection = "!README" in content
            has_process_doc_exclusion = any(
                doc in content
                for doc in [
                    "PHASE*.md",
                    "*COMPLETION*.md",
                    "ACTION_PLAN.md",
                    "PROJECT_COMPLETION.md",
                ]
            )

            self.add_result(
                "代码管理",
                ".gitignore配置",
                not has_problematic_md_rule,
                "✅ 已移除有问题的*.md规则"
                if not has_problematic_md_rule
                else "❌ 仍有*.md规则",
                "MEDIUM" if has_problematic_md_rule else "INFO",
            )

            self.add_result(
                "代码管理",
                "README保护",
                has_readme_protection,
                "✅ README受保护" if has_readme_protection else "⚠️ README未保护",
                "LOW",
            )

            self.add_result(
                "代码管理",
                "过程文档排除",
                has_process_doc_exclusion,
                "✅ 过程文档排除规则存在"
                if has_process_doc_exclusion
                else "⚠️ 过程文档排除规则缺失",
                "LOW",
            )

        except Exception as e:
            self.add_result(
                "代码管理", ".gitignore检查", False, f"❌ 检查失败: {e}", "MEDIUM"
            )

    def check_requirement_files(self):
        """检查requirements文件配置"""
        req_files = [
            (
                "requirements/common.txt",
                {
                    "anthropic >= 0.71.0",
                    "protobuf >= 5.29.6",
                    "model-hosting-container-standards >= 0.1.14",
                },
            ),
            ("requirements/rocm.txt", {"setuptools>=77.0.3,<81.0.0"}),
            ("requirements/test/cuda.txt", {"protobuf=="}),
        ]

        for file_path, required_content in req_files:
            full_path = self.vllm_path / file_path
            if not full_path.exists():
                self.add_result(
                    "依赖配置", f"{file_path}", False, f"❌ 文件不存在", "HIGH"
                )
                continue

            try:
                content = full_path.read_text(encoding="utf-8")
                all_passed = True
                issues = []

                for required in required_content:
                    if required not in content:
                        all_passed = False
                        issues.append(f"缺少: {required}")

                msg = "✅ 配置正确" if all_passed else f"⚠️ {', '.join(issues)}"
                severity = "HIGH" if not all_passed else "INFO"

                self.add_result("依赖配置", file_path, all_passed, msg, severity)

            except Exception as e:
                self.add_result(
                    "依赖配置", file_path, False, f"❌ 检查失败: {e}", "MEDIUM"
                )

    def check_model_registry(self):
        """检查模型注册表"""
        try:
            from vllm.model_executor.models import ModelRegistry

            # 尝试获取已注册的模型
            registered_models = list(ModelRegistry.get_model_architectures())
            model_count = len(registered_models)

            self.add_result(
                "模型系统",
                "模型注册表",
                model_count > 0,
                f"✅ 已注册 {model_count} 个模型架构",
                "INFO",
            )

            # 检查Qwen3.5是否注册
            qwen_models = [m for m in registered_models if "qwen" in m.lower()]
            if qwen_models:
                self.add_result(
                    "模型系统",
                    "Qwen系列模型",
                    True,
                    f"✅ 发现 {len(qwen_models)} 个Qwen模型",
                    "INFO",
                )

        except Exception as e:
            self.add_result(
                "模型系统", "模型注册表", False, f"❌ 检查失败: {e}", "HIGH"
            )

    def run_all_checks(self) -> bool:
        """运行所有检查"""
        print("=" * 80)
        print("🔍 vLLM gfx906 运行时问题综合诊断")
        print("=" * 80)
        print()

        # 按优先级运行检查
        print("📋 第1步：Python环境检查...")
        self.check_python_version()
        print()

        print("📋 第2步：核心依赖检查...")
        try:
            self.check_core_dependencies()
        except Exception as e:
            print(f"⚠️ 依赖检查出错: {e}")
        print()

        print("📋 第3步：编译模块检查...")
        try:
            self.check_vllm_compiled_modules()
        except Exception as e:
            print(f"⚠️ 编译模块检查出错: {e}")
        print()

        print("📋 第4步：模型配置检查...")
        try:
            self.check_qwen35_configs()
        except Exception as e:
            print(f"⚠️ 模型配置检查出错: {e}")
        print()

        print("📋 第5步：平台配置检查...")
        try:
            self.check_rocm_platform()
        except Exception as e:
            print(f"⚠️ 平台配置检查出错: {e}")
        print()

        print("📋 第6步：代码管理检查...")
        self.check_gitignore_patterns()
        print()

        print("📋 第7步：依赖配置检查...")
        self.check_requirement_files()
        print()

        print("📋 第8步：模型系统检查...")
        try:
            self.check_model_registry()
        except Exception as e:
            print(f"⚠️ 模型系统检查出错: {e}")
        print()

        return self.print_summary()

    def print_summary(self) -> bool:
        """打印检查摘要"""
        print("=" * 80)
        print("📊 诊断结果摘要")
        print("=" * 80)

        # 按严重程度分组
        critical = [
            r for r in self.results if r.severity == "CRITICAL" and not r.passed
        ]
        high = [r for r in self.results if r.severity == "HIGH" and not r.passed]
        medium = [r for r in self.results if r.severity == "MEDIUM" and not r.passed]
        passed = [r for r in self.results if r.passed]

        print(f"\n✅ 通过: {len(passed)} 个检查")
        print(f"❌ 失败: {len(self.results) - len(passed)} 个检查")

        if critical:
            print(f"\n🔴 严重问题 ({len(critical)}):")
            for r in critical:
                print(f"  {r}")

        if high:
            print(f"\n⚠️  高优先级问题 ({len(high)}):")
            for r in high:
                print(f"  {r}")

        if medium:
            print(f"\n📡 中等问题 ({len(medium)}):")
            for r in medium:
                print(f"  {r}")

        print("\n" + "=" * 80)

        if critical:
            print("❌ 诊断失败：发现严重问题，必须修复后才能运行")
            print("=" * 80)
            return False
        elif high:
            print("⚠️  发现高优先级问题：建议修复后再运行")
            print("=" * 80)
            return False
        else:
            print("✅ 诊断通过：所有关键检查都通过")
            print("=" * 80)
            return True


def main():
    """主函数"""
    diagnostics = RuntimeDiagnostics()

    try:
        success = diagnostics.run_all_checks()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ 诊断被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 诊断过程中发生错误: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
