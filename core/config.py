
# CONFIGURATION SPÉCIFIQUE M4
if platform.processor() == 'arm':
    os.environ['TF_METAL_ENABLED'] = '1'
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    torch.set_float32_matmul_precision('high')
