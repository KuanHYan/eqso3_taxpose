import torch
from taxpose.nets.transformer_flow import create_network
import hydra
from torchview import draw_graph
import graphviz
graphviz.set_jupyter_format('png')

@hydra.main(version_base="1.1", config_path="./configs", config_name="train_ndf")
def main(cfg):
    model = create_network(cfg.model)
    checkpoint = torch.load(cfg.resume_ckpt, map_location='cpu')['state_dict']
    fix = 'model.'
    checkpoint = {k[len(fix):]: v for k, v in checkpoint.items() if k.startswith(fix)}
    model.load_state_dict(checkpoint)
    model.eval()
    # 创建一个虚拟输入
    dummy_input_1 = torch.randn(1, 1024, 3)
    dummy_input_2 = torch.randn(1, 1024, 3)
    # 4. 导出为 ONNX 格式
    # torch.onnx.export(model, (dummy_input_1, dummy_input_2), f"/home/yan/pose_estimation/taxpose/vis_taxpose.onnx")
    # print(f"ONNX model exported to {cfg.wandb.name}.onnx")
    model_graph = draw_graph(model,
                             input_data=(dummy_input_1, dummy_input_2),
                             device='meta',
                             filename='test_graph',
                             save_graph=True)
    model_graph.visual_graph

if __name__ == "__main__":
    main()