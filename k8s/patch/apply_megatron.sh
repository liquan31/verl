set -x

cwd=$(dirname $(realpath $0))
echo "dealing $cwd"
cd $cwd

\cp ./megatron.patch/0.12.1/Megatron-LM/megatron/core/transformer/multi_token_prediction.py /opt/Megatron-LM/megatron/core/transformer/multi_token_prediction.py

echo -e "/033[32mApplied megatron MTP done./033[0m"

\cp ./megatron.patch/0.12.1/Megatron-LM/megatron/core/transformer/transformer_block.py /opt/Megatron-LM/megatron/core/transformer/transformer_block.py

\cp ./megatron.patch/0.12.1/Megatron-LM/megatron/core/transformer/routing_replay.py /opt/Megatron-LM/megatron/core/transformer/routing_replay.py

\cp ./megatron.patch/0.12.1/Megatron-LM/megatron/core/transformer/moe/router.py /opt/Megatron-LM/megatron/core/transformer/moe/router.py

\cp ./megatron.patch/0.12.1/Megatron-LM/megatron/core/transformer/moe/moe_utils.py /opt/Megatron-LM/megatron/core/transformer/moe/moe_utils.py

echo -e "/033[32mApplied megatron routing replay done./033[0m"

set +x
