
set -x # 开启调试模式
cwd=$(dirname $(realpath $0))
echo "dealing $cwd"
cd $cwd

# vllm_version=$(cat /opt/vllm/vllm/_version.py | grep 'version =' | awk '{print $5}')
# vllm_version=${vllm_version//\'}
# echo -e "\033[1;33m vllm_version: ${vllm_version}\033[0m"
vllm_version='0.11.0'
if [[ ${vllm_version} == '0.9.1' ]];then
  # rm -f /opt/vllm/vllm/model_executor/models/deepseek_v2.py
  # cp -f /home/code/verl/k8s/patch/0827/deepseek_v2.py /opt/vllm/vllm/model_executor/models/deepseek_v2.py
  echo -e "\033[1;33mApplied VLLM-ASCEND ${vllm_version}! 这个版本的patch不在当前代码仓中！\033[0m"
fi
if [[ ${vllm_version} == '0.10.0' ]];then
  \cp -f ./vllm.patch/0.10.0/vllm-ascend/vllm_ascend/models/deepseek_v2.py /opt/vllm-ascend/vllm_ascend/models/deepseek_v2.py
  \cp -f ./vllm.patch/0.10.0/vllm-ascend/vllm_ascend/attention/mla_v1.py /opt/vllm-ascend/vllm_ascend/attention/mla_v1.py
  \cp -f ./vllm.patch/0.10.0/vllm-ascend/vllm_ascend/ops/rotary_embedding.py /opt/vllm-ascend/vllm_ascend/ops/rotary_embedding.py

  echo -e "\033[32mApplied VLLM-ASCEND ${vllm_version}!\033[0m"
fi
if [[ ${vllm_version} == '0.11.0' ]];then
  rm -f /opt/vllm/vllm/model_executor/layers/rotary_embedding/base.py
  cp -f ./vllm.patch/0.11.0/vllm/vllm/model_executor/layers/rotary_embedding/base.py /opt/vllm/vllm/model_executor/layers/rotary_embedding/base.py

  rm -f /opt/vllm-ascend/vllm_ascend/ops/common_fused_moe.py
  cp -f ./vllm.patch/0.11.0/r3/vllm-ascend/vllm_ascend/ops/common_fused_moe.py /opt/vllm-ascend/vllm_ascend/ops/common_fused_moe.py

  rm -f /opt/vllm-ascend/vllm_ascend/worker/model_runner_v1.py
  cp -f ./vllm.patch/0.11.0/r3/vllm-ascend/vllm_ascend/worker/model_runner_v1.py /opt/vllm-ascend/vllm_ascend/worker/model_runner_v1.py

  rm -f /opt/vllm/vllm/config/__init__.py
  cp -f ./vllm.patch/0.11.0/r3/vllm/vllm/config/__init__.py /opt/vllm/vllm/config/__init__.py

  rm -f /opt/vllm/vllm/config/model.py
  cp -f ./vllm.patch/0.11.0/r3/vllm/vllm/config/model.py /opt/vllm/vllm/config/model.py

  rm -f /opt/vllm/vllm/engine/arg_utils.py
  cp -f ./vllm.patch/0.11.0/r3/vllm/vllm/engine/arg_utils.py /opt/vllm/vllm/engine/arg_utils.py

  rm -f /opt/vllm/vllm/entrypoints/llm.py
  cp -f ./vllm.patch/0.11.0/r3/vllm/vllm/entrypoints/llm.py /opt/vllm/vllm/entrypoints/llm.py

  cp -f ./vllm.patch/0.11.0/r3/vllm/vllm/model_executor/layers/fused_moe/routed_experts_capturer.py /opt/vllm/vllm/model_executor/layers/fused_moe/routed_experts_capturer.py

  rm -f /opt/vllm/vllm/v1/core/sched/scheduler.py
  cp -f ./vllm.patch/0.11.0/r3/vllm/vllm/v1/core/sched/scheduler.py /opt/vllm/vllm/v1/core/sched/scheduler.py

  rm -f /opt/vllm/vllm/v1/engine/__init__.py
  cp -f ./vllm.patch/0.11.0/r3/vllm/vllm/v1/engine/__init__.py /opt/vllm/vllm/v1/engine/__init__.py

  rm -f /opt/vllm/vllm/v1/engine/output_processor.py
  cp -f ./vllm.patch/0.11.0/r3/vllm/vllm/v1/engine/output_processor.py /opt/vllm/vllm/v1/engine/output_processor.py

  rm -f /opt/vllm/vllm/outputs.py
  cp -f ./vllm.patch/0.11.0/r3/vllm/vllm/outputs.py /opt/vllm/vllm/outputs.py

  echo -e "\033[1;33mApplied VLLM-ASCEND ${vllm_version}! patch成功\033[0m"
fi
set +x # 关闭调试模式