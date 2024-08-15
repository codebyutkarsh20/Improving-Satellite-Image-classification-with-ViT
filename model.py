import torch
import timm
from torch import nn
from einops import rearrange
from lora import LoRA_ViT_timm, LoRA_ViT_timm_mod
from adapter import Adapter_ViT
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn.parameter import Parameter

class LORAModel(LoRA_ViT_timm):
    def __init__(self, r = 4,num_classes = 30, pretrained = True, freeze = True, layer = -1):
        self.layer = layer
        super().__init__(vit_model=timm.create_model('vit_small_patch16_224',num_classes = 0, pretrained=pretrained), r=r)
        self.model = self.lora_vit
        del self.lora_vit
        self.head = nn.Linear(self.model.embed_dim, num_classes)
        self.num_heads = self.model.blocks[layer].attn.num_heads

    def forward(self, inp1, inp2=None):
        if inp2 is None:
            inp2 = torch.zeros_like(inp1)
        feats = []
        # Using multiple layers
        B,C,H,W = inp1.shape

        def hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            q = qkv[0]
            k = qkv[1]
            v = qkv[2]
            scale = q.shape[-1] ** -0.5
            attn = (q@k.transpose(-2, -1))*scale
            attn = attn.softmax(dim=-1)
            feats.append(attn)

        id = self.model.blocks[self.layer].attn.register_forward_hook(hook)

        output_1 = self.head(self.model(inp1))
        # removing the cls token
        self_attn_1 = feats[-1][:,:,1:,1:]
        dim = int(self_attn_1.shape[-2]**0.5)
        # self_attn_1 = self_attn_1.reshape(B,self.num_heads,dim,dim,dim,dim) # b,12,14,14,14,14
        self_attn_1 = rearrange(self_attn_1, 'b h (h1 w1) (h2 w2) -> b h h1 w1 h2 w2', h2 = dim, h1 = dim)


        output_2 = self.head(self.model(inp2))
        # removing the cls token
        self_attn_2 = feats[-1][:,:,1:,1:]
        dim = int(self_attn_2.shape[-2]**0.5)
        self_attn_2 = rearrange(self_attn_2, 'b h (h1 w1) (h2 w2) -> b h h1 w1 h2 w2', h2 = dim, h1 = dim)
        # print(self_attn_2.shape)

        id.remove()
        # return output_1
        return self_attn_1, output_1, self_attn_2, output_2

    def triplet_forward(self, anc, pos, neg):
        return self.model(anc),self.model(pos),self.model(neg)

    def save_model(self, epoch,exp, latest = True):
        extension = 'latest' if latest else 'best'
        #first saving the head
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """
        # _in = self.head.in_features
        # _out = self.head.out_features
        # fc_tensors = {f"fc_{_in}in_{_out}out": self.head.weight}
        # save_file(fc_tensors, f'{exp}/classifier_epoch_{epoch}.safetensors')

        # # saving the lora weights
        # self.save_lora_parameters(f'{exp}/lora_epoch_{epoch}.safetensors')
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        filename = f'{exp}/lora_epoch_{extension}.safetensors'

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}

        _in = self.head.in_features
        _out = self.head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.head.weight}

        merged_dict = {**a_tensors, **b_tensors, **fc_tensors, 'epoch':torch.tensor(epoch)}
        save_file(merged_dict, filename)

    def load_model(self,exp,  latest = True):
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """
        extension = 'latest' if latest else 'best'
        # _in = self.head.in_features
        # _out = self.head.out_features
        # with safe_open(f'{exp}/classifier_epoch_{epoch}.safetensors', framework="pt") as f:
        #     saved_key = f"fc_{_in}in_{_out}out"
        #     try:
        #         saved_tensor = f.get_tensor(saved_key)
        #         self.head.weight = Parameter(saved_tensor)
        #     except ValueError:
        #         print("this fc weight is not for this model")

        # self.load_lora_parameters(f'{exp}/lora_epoch_{epoch}.safetensors')

        filename = f'{exp}/lora_epoch_{extension}.safetensors'
        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)

            _in = self.head.in_features
            _out = self.head.out_features
            saved_key = f"fc_{_in}in_{_out}out"
            epoch = f.get_tensor('epoch')
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.head.weight = Parameter(saved_tensor)
                print('Model loaded successfully')
            except ValueError:
                print("this fc weight is not for this model")

        return epoch.item()

class LORAModelMod(LoRA_ViT_timm_mod):
    def __init__(self, r = 4,num_classes = 30, pretrained = True, freeze = True, layer = -1):
        self.layer = layer
        super().__init__(vit_model=timm.create_model('vit_small_patch16_224',num_classes = 0, pretrained=pretrained), r=r)
        self.model = self.lora_vit
        del self.lora_vit
        self.head = nn.Linear(self.model.embed_dim, num_classes)
        self.num_heads = self.model.blocks[layer].attn.num_heads

    def forward(self, inp1, inp2):
        feats = []
        # Using multiple layers
        B,C,H,W = inp1.shape

        def hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            q = qkv[0]
            k = qkv[1]
            v = qkv[2]
            scale = q.shape[-1] ** -0.5
            attn = (q@k.transpose(-2, -1))*scale
            attn = attn.softmax(dim=-1)
            feats.append(attn)

        id = self.model.blocks[self.layer].attn.register_forward_hook(hook)

        output_1 = self.head(self.model(inp1))
        # removing the cls token
        self_attn_1 = feats[-1][:,:,1:,1:]
        dim = int(self_attn_1.shape[-2]**0.5)
        # self_attn_1 = self_attn_1.reshape(B,self.num_heads,dim,dim,dim,dim) # b,12,14,14,14,14
        self_attn_1 = rearrange(self_attn_1, 'b h (h1 w1) (h2 w2) -> b h h1 w1 h2 w2', h2 = dim, h1 = dim)


        output_2 = self.head(self.model(inp2))
        # removing the cls token
        self_attn_2 = feats[-1][:,:,1:,1:]
        dim = int(self_attn_2.shape[-2]**0.5)
        self_attn_2 = rearrange(self_attn_2, 'b h (h1 w1) (h2 w2) -> b h h1 w1 h2 w2', h2 = dim, h1 = dim)
        # print(self_attn_2.shape)

        id.remove()

        return self_attn_1, output_1, self_attn_2, output_2

    def triplet_forward(self, anc, pos, neg):
        return self.model(anc),self.model(pos),self.model(neg)

    def save_model(self, epoch,exp, latest = True):
        extension = 'latest' if latest else 'best'
        #first saving the head
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """
        # _in = self.head.in_features
        # _out = self.head.out_features
        # fc_tensors = {f"fc_{_in}in_{_out}out": self.head.weight}
        # save_file(fc_tensors, f'{exp}/classifier_epoch_{epoch}.safetensors')

        # # saving the lora weights
        # self.save_lora_parameters(f'{exp}/lora_epoch_{epoch}.safetensors')
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        filename = f'{exp}/lora_epoch_{extension}.safetensors'

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}

        _in = self.head.in_features
        _out = self.head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.head.weight}

        merged_dict = {**a_tensors, **b_tensors, **fc_tensors, 'epoch':torch.tensor(epoch)}
        save_file(merged_dict, filename)

    def load_model(self,exp, latest = True):
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """
        extension = 'latest' if latest else 'best'
        # _in = self.head.in_features
        # _out = self.head.out_features
        # with safe_open(f'{exp}/classifier_epoch_{epoch}.safetensors', framework="pt") as f:
        #     saved_key = f"fc_{_in}in_{_out}out"
        #     try:
        #         saved_tensor = f.get_tensor(saved_key)
        #         self.head.weight = Parameter(saved_tensor)
        #     except ValueError:
        #         print("this fc weight is not for this model")

        # self.load_lora_parameters(f'{exp}/lora_epoch_{epoch}.safetensors')

        filename = f'{exp}/lora_epoch_{extension}.safetensors'
        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)

            _in = self.head.in_features
            _out = self.head.out_features
            saved_key = f"fc_{_in}in_{_out}out"
            epoch = f.get_tensor('epoch')
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.head.weight = Parameter(saved_tensor)
                print('Model loaded successfully')
            except ValueError:
                print("this fc weight is not for this model")

        return epoch.item()

class AdapterModel(Adapter_ViT):
    def __init__(self, r = 4,num_classes = 30, pretrained = True, freeze = True, layer = -1):
        super().__init__(vit_model=timm.create_model('vit_small_patch16_224',num_classes = 0, pretrained=pretrained), num_classes = num_classes)

        self.layer = layer
        self.num_classes = num_classes
        self.model = self.backbone
        del self.backbone


    def forward(self, inp1, inp2):
        feats = []
        # Using multiple layers
        B,C,H,W = inp1.shape

        def hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            q = qkv[0]
            k = qkv[1]
            v = qkv[2]
            scale = q.shape[-1] ** -0.5
            attn = (q@k.transpose(-2, -1))*scale
            attn = attn.softmax(dim=-1)
            feats.append(attn)

        id = self.model.blocks[self.layer].attn.register_forward_hook(hook)

        output_1 = self.adapter(self.model(inp1))
        # removing the cls token
        self_attn_1 = feats[-1][:,:,1:,1:]
        dim = int(self_attn_1.shape[-2]**0.5)
        # self_attn_1 = self_attn_1.reshape(B,self.num_heads,dim,dim,dim,dim) # b,12,14,14,14,14
        self_attn_1 = rearrange(self_attn_1, 'b h (h1 w1) (h2 w2) -> b h h1 w1 h2 w2', h2 = dim, h1 = dim)


        output_2 = self.adapter(self.model(inp2))
        # removing the cls token
        self_attn_2 = feats[-1][:,:,1:,1:]
        dim = int(self_attn_2.shape[-2]**0.5)
        self_attn_2 = rearrange(self_attn_2, 'b h (h1 w1) (h2 w2) -> b h h1 w1 h2 w2', h2 = dim, h1 = dim)
        # print(self_attn_2.shape)

        id.remove()

        return self_attn_1, output_1, self_attn_2, output_2

    def triplet_forward(self, anc, pos, neg):
        return self.model(anc),self.model(pos),self.model(neg)

    def save_model(self, epoch,exp, latest = True):
        if latest:
            torch.save(
                {
                    'head':self.adapter.state_dict(),
                    'epoch':epoch,
                },
                f'{exp}/checkpoint_latest.pth'
            )
        else:
            torch.save(
                {
                    'head':self.adapter.state_dict(),
                    'epoch':epoch,
                },
                f'{exp}/checkpoint_best.pth'
            )

    def load_model(self, exp,  latest = True):
        if latest:
            ckpt = torch.load(f'{exp}/checkpoint_latest.pth', map_location='cpu')
        else:
            ckpt = torch.load(f'{exp}/checkpoint_best.pth', map_location='cpu')

        self.adapter.load_state_dict(ckpt['head'])
        return ckpt['epoch']

class Model(nn.Module):
    def __init__(self,num_classes = 30, pretrained = True, freeze = True, layer = -1):
        super().__init__()
        self.layer = layer
        # self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained = pretrained)
        self.model = timm.create_model('vit_small_patch16_224',num_classes = 0, pretrained=pretrained)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            # for p in range(len(self.model.blocks)):
            #     for param in self.model.blocks[p].attn.qkv.parameters():
            #         param.requires_grad = True
            #     for param in self.model.blocks[p].attn.parameters():
            #         param.requires_grad = True


        self.head = nn.Linear(self.model.embed_dim, num_classes)
        self.num_heads = self.model.blocks[layer].attn.num_heads

    def forward(self, inp1, inp2):
        feats = []
        # Using multiple layers
        B,C,H,W = inp1.shape

        def hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            q = qkv[0]
            k = qkv[1]
            v = qkv[2]
            scale = q.shape[-1] ** -0.5
            attn = (q@k.transpose(-2, -1))*scale
            attn = attn.softmax(dim=-1)
            feats.append(attn)

        id = self.model.blocks[self.layer].attn.register_forward_hook(hook)

        output_1 = self.head(self.model(inp1))
        # removing the cls token
        self_attn_1 = feats[-1][:,:,1:,1:]
        dim = int(self_attn_1.shape[-2]**0.5)
        # self_attn_1 = self_attn_1.reshape(B,self.num_heads,dim,dim,dim,dim) # b,12,14,14,14,14
        self_attn_1 = rearrange(self_attn_1, 'b h (h1 w1) (h2 w2) -> b h h1 w1 h2 w2', h2 = dim, h1 = dim)


        output_2 = self.head(self.model(inp2))
        # removing the cls token
        self_attn_2 = feats[-1][:,:,1:,1:]
        dim = int(self_attn_2.shape[-2]**0.5)
        self_attn_2 = rearrange(self_attn_2, 'b h (h1 w1) (h2 w2) -> b h h1 w1 h2 w2', h2 = dim, h1 = dim)
        # print(self_attn_2.shape)

        id.remove()

        return self_attn_1, output_1, self_attn_2, output_2

    def triplet_forward(self, anc, pos, neg):
        return self.model(anc),self.model(pos),self.model(neg)

    def save_model(self, epoch, exp, latest = True):
        if latest:
            torch.save(
                {
                    'model':self.model.state_dict(),
                    'head':self.head.state_dict(),
                    'epoch':epoch,
                },
                f'{exp}/checkpoint_latest.pth'
            )
        else:
            torch.save(
                {
                    'model':self.model.state_dict(),
                    'head':self.head.state_dict(),
                    'epoch':epoch,
                },
                f'{exp}/checkpoint_best.pth'
            )

    def load_model(self, exp,  latest = True):
        if latest:
            ckpt = torch.load(f'{exp}/checkpoint_latest.pth', map_location='cpu')
        else:
            ckpt = torch.load(f'{exp}/checkpoint_best.pth', map_location='cpu')

        self.model.load_state_dict(ckpt['model'])
        self.head.load_state_dict(ckpt['head'])
        return ckpt['epoch']


class VPTModel(nn.Module):
    def __init__ (self, num_tokens = 2,num_classes = 30, pretrained = True, freeze = True, layer = -1):
        super().__init__()
        self.num_tokens = num_tokens
        self.layer = layer
        self.model = timm.create_model('vit_small_patch16_224',num_classes = 0, pretrained=pretrained)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.norm_pre = VPT(num_tokens)
        self.head = nn.Linear(self.model.embed_dim, num_classes)
        self.num_heads = self.model.blocks[layer].attn.num_heads
    
    def forward(self, inp1, inp2):
        feats = []
        # Using multiple layers
        B,C,H,W = inp1.shape

        def hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            q = qkv[0]
            k = qkv[1]
            v = qkv[2]
            scale = q.shape[-1] ** -0.5
            attn = (q@k.transpose(-2, -1))*scale
            attn = attn.softmax(dim=-1)
            feats.append(attn)

        id = self.model.blocks[self.layer].attn.register_forward_hook(hook)

        output_1 = self.head(self.model(inp1))
        # removing the cls token
        self_attn_1 = feats[-1][:,:,1+self.num_tokens:,1+self.num_tokens:]
        dim = int(self_attn_1.shape[-2]**0.5)
        # self_attn_1 = self_attn_1.reshape(B,self.num_heads,dim,dim,dim,dim) # b,12,14,14,14,14
        self_attn_1 = rearrange(self_attn_1, 'b h (h1 w1) (h2 w2) -> b h h1 w1 h2 w2', h2 = dim, h1 = dim)


        output_2 = self.head(self.model(inp2))
        # removing the cls token
        self_attn_2 = feats[-1][:,:,1+self.num_tokens:,1+self.num_tokens:]
        dim = int(self_attn_2.shape[-2]**0.5)
        self_attn_2 = rearrange(self_attn_2, 'b h (h1 w1) (h2 w2) -> b h h1 w1 h2 w2', h2 = dim, h1 = dim)
        # print(self_attn_2.shape)

        id.remove()

        return self_attn_1, output_1, self_attn_2, output_2

    def triplet_forward(self, anc, pos, neg):
        return self.model(anc),self.model(pos),self.model(neg)

    def save_model(self, epoch, exp, latest = True):
        if latest:
            torch.save(
                {
                    'model':self.model.state_dict(),
                    'head':self.head.state_dict(),
                    'epoch':epoch,
                },
                f'{exp}/checkpoint_latest.pth'
            )
        else:
            torch.save(
                {
                    'model':self.model.state_dict(),
                    'head':self.head.state_dict(),
                    'epoch':epoch,
                },
                f'{exp}/checkpoint_best.pth'
            )

    def load_model(self, exp,  latest = True):
        if latest:
            ckpt = torch.load(f'{exp}/checkpoint_latest.pth', map_location='cpu')
        else:
            ckpt = torch.load(f'{exp}/checkpoint_best.pth', map_location='cpu')

        self.model.load_state_dict(ckpt['model'])
        self.head.load_state_dict(ckpt['head'])
        return ckpt['epoch']

class VPTLORAModel(LoRA_ViT_timm):
    def __init__(self,num_tokens, r = 4,num_classes = 30, pretrained = True, freeze = True, layer = -1):
        self.layer = layer
        self.num_tokens = num_tokens
        super().__init__(vit_model=timm.create_model('vit_small_patch16_224',num_classes = 0, pretrained=pretrained), r=r)
        self.model = self.lora_vit
        del self.lora_vit
        self.model.norm_pre = VPT(num_tokens)
        self.head = nn.Linear(self.model.embed_dim, num_classes)
        self.num_heads = self.model.blocks[layer].attn.num_heads

    def forward(self, inp1, inp2=None):
        if inp2 is None:
            inp2 = torch.zeros_like(inp1)
        feats = []
        # Using multiple layers
        B,C,H,W = inp1.shape

        def hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            q = qkv[0]
            k = qkv[1]
            v = qkv[2]
            scale = q.shape[-1] ** -0.5
            attn = (q@k.transpose(-2, -1))*scale
            attn = attn.softmax(dim=-1)
            feats.append(attn)

        id = self.model.blocks[self.layer].attn.register_forward_hook(hook)

        output_1 = self.head(self.model(inp1))
        # removing the cls token
        self_attn_1 = feats[-1][:,:,1+self.num_tokens:,1+self.num_tokens:]
        dim = int(self_attn_1.shape[-2]**0.5)
        # self_attn_1 = self_attn_1.reshape(B,self.num_heads,dim,dim,dim,dim) # b,12,14,14,14,14
        self_attn_1 = rearrange(self_attn_1, 'b h (h1 w1) (h2 w2) -> b h h1 w1 h2 w2', h2 = dim, h1 = dim)


        output_2 = self.head(self.model(inp2))
        # removing the cls token
        self_attn_2 = feats[-1][:,:,1+self.num_tokens:,1+self.num_tokens:]
        dim = int(self_attn_2.shape[-2]**0.5)
        self_attn_2 = rearrange(self_attn_2, 'b h (h1 w1) (h2 w2) -> b h h1 w1 h2 w2', h2 = dim, h1 = dim)
        # print(self_attn_2.shape)

        id.remove()
        # return output_1
        return self_attn_1, output_1, self_attn_2, output_2

    def triplet_forward(self, anc, pos, neg):
        return self.model(anc),self.model(pos),self.model(neg)

    def save_model(self, epoch,exp, latest = True):
        extension = 'latest' if latest else 'best'
        #first saving the head
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """
        # _in = self.head.in_features
        # _out = self.head.out_features
        # fc_tensors = {f"fc_{_in}in_{_out}out": self.head.weight}
        # save_file(fc_tensors, f'{exp}/classifier_epoch_{epoch}.safetensors')

        # # saving the lora weights
        # self.save_lora_parameters(f'{exp}/lora_epoch_{epoch}.safetensors')
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        filename = f'{exp}/lora_epoch_{extension}.safetensors'
        # filename = 'trial.safetensors'

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}

        _in = self.head.in_features
        _out = self.head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.head.weight}

        vpt_tensors = {"vpt_state_dict":self.model.norm_pre.state_dict()['prompt_embeddings']}

        merged_dict = {**a_tensors, **b_tensors, **fc_tensors,**vpt_tensors, 'epoch':torch.tensor(epoch)}
        save_file(merged_dict, filename)

    def load_model(self,exp,  latest = True):
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """
        extension = 'latest' if latest else 'best'
        # _in = self.head.in_features
        # _out = self.head.out_features
        # with safe_open(f'{exp}/classifier_epoch_{epoch}.safetensors', framework="pt") as f:
        #     saved_key = f"fc_{_in}in_{_out}out"
        #     try:
        #         saved_tensor = f.get_tensor(saved_key)
        #         self.head.weight = Parameter(saved_tensor)
        #     except ValueError:
        #         print("this fc weight is not for this model")

        # self.load_lora_parameters(f'{exp}/lora_epoch_{epoch}.safetensors')

        filename = f'{exp}/lora_epoch_{extension}.safetensors'
        # filename = 'trial.safetensors'

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)

            _in = self.head.in_features
            _out = self.head.out_features
            saved_key = f"fc_{_in}in_{_out}out"
            epoch = f.get_tensor('epoch')
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.head.weight = Parameter(saved_tensor)
                vpt_state_dict = {'prompt_embeddings':f.get_tensor('vpt_state_dict')}
                self.model.norm_pre.load_state_dict(vpt_state_dict)
                print('Model loaded successfully')
            except ValueError:
                print("this fc weight is not for this model")

        return epoch.item()

class VPT(nn.Module):
    def __init__(self, num_tokens):
        super().__init__()
        # val = math.sqrt(6. / float(3 * reduce(mul, 16, 1) + 384))  # noqa
        self.num_tokens = num_tokens
        self.prompt_embeddings = nn.Parameter(torch.zeros(
            1, num_tokens, 384))
        # xavier_uniform initialization
        # nn.init.uniform_(self.prompt_embeddings.data, -val, val)
    
    def forward(self, x):
        B = x.shape[0]
        out = torch.cat((
                x[:, :1, :],
                self.prompt_embeddings.expand(B, -1, -1),
                x[:, 1:, :]
            ), dim=1)
        return out

