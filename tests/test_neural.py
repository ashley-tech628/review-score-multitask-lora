import importlib.util
import unittest

AVAILABLE=all(importlib.util.find_spec(m) for m in ['torch','transformers','peft','safetensors'])


@unittest.skipUnless(AVAILABLE,'Optional neural dependencies unavailable')
class NeuralTests(unittest.TestCase):
    def test_masked_loss_and_gradients(self):
        import torch
        from reviewscore.neural import masked_mse
        p=torch.zeros(2,5,requires_grad=True)
        y=torch.ones(2,5);y[0,0]=float('nan')
        loss=masked_mse(p,y);loss.backward()
        self.assertEqual(loss.item(),1.)
        self.assertEqual(p.grad[0,0].item(),0.)
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_offline_tiny_model_has_trainable_head(self):
        import torch
        from transformers import DistilBertConfig, DistilBertModel
        from peft import LoraConfig, get_peft_model
        from reviewscore.neural import ScoreModel
        base=DistilBertModel(DistilBertConfig(vocab_size=30,dim=16,hidden_dim=32,n_layers=1,n_heads=2))
        model=ScoreModel(get_peft_model(base,LoraConfig(r=2,target_modules=['q_lin','v_lin'])))
        self.assertTrue(all(p.requires_grad for p in model.head.parameters()))
        output=model(torch.ones(2,4,dtype=torch.long),torch.ones(2,4,dtype=torch.long))
        self.assertEqual(tuple(output.shape),(2,5))
        self.assertTrue(((output>=0)&(output<=1)).all())

    def test_complete_bundle_roundtrip_offline(self):
        import tempfile
        from pathlib import Path
        import torch
        from transformers import DistilBertConfig, DistilBertModel, BertTokenizerFast
        from reviewscore.neural import create, save_bundle, load_bundle
        with tempfile.TemporaryDirectory() as d:
            base=Path(d)/'base';base.mkdir()
            vocab=['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','good','aroma']
            (base/'vocab.txt').write_text('\n'.join(vocab),encoding='utf-8')
            BertTokenizerFast(vocab_file=str(base/'vocab.txt')).save_pretrained(base)
            DistilBertModel(DistilBertConfig(vocab_size=len(vocab),dim=16,hidden_dim=32,n_layers=1,n_heads=2)).save_pretrained(base)
            model,tokenizer=create(str(base),local_only=True);model.eval()
            batch=tokenizer(['good aroma'],return_tensors='pt')
            args={k:v for k,v in batch.items() if k in ['input_ids','attention_mask']}
            with torch.no_grad():original=model(**args)
            save_bundle(model,tokenizer,Path(d)/'bundle',str(base))
            restored,_=load_bundle(Path(d)/'bundle',local_only=True)
            with torch.no_grad():torch.testing.assert_close(original,restored(**args))


if __name__=='__main__':unittest.main()
