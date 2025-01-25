import argparse
from itertools import islice
from pathlib import Path
from typing import Iterable, List, Optional

import h5py
import torch
from pytorch_nndct.apis import dump_xmodel, torch_quantizer
from tqdm import tqdm

from sml_tutorials_ml_deployment.model import Unet

# These are derived from the model training and deployment prepration processes
DEFAULT_MODEL_INPUT_SIZE = [3, 512, 512]
DEFAULT_MODEL_NUM_CLASSES = 7


def batched(iterable: Iterable, batch_size: int):
    """Batch data from the iterable into tuples of length batch_size: batched('ABCDEFG', 3) -> ABC DEF G."""
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, batch_size)):
        yield batch


def calibrate(
    model: torch.nn.Module,
    input_shape: List[int],
    batch_size: int,
    data_h5_path: Path,
    output_dir: Path,
    samples_num_limit: Optional[int] = None,
):
    # Initialize the quantizer by passing dummy data with the same shape as the model input
    dummy_input = torch.randn(batch_size, *input_shape)
    quantizer = torch_quantizer("calib", model, (dummy_input), output_dir=str(output_dir))
    quant_model = quantizer.quant_model

    # Perform calibration based on train samples
    with h5py.File(data_h5_path, "r") as f_in:
        sample_names = list(f_in["calibration"].keys())[:samples_num_limit]
        for names_batch in tqdm(batched(sample_names, batch_size)):
            input_batch = torch.stack([torch.as_tensor(f_in[f"calibration/{name}"][:]) for name in names_batch])
            quant_model(input_batch)

    quantizer.export_quant_config()


def test(
    model: torch.nn.Module,
    input_shape: List[int],
    data_h5_path: Path,
    output_dir: Path,
    test_samples: Path,
    samples_num_limit: Optional[int] = None,
):
    # Initialize the quantizer by passing dummy data with the same shape as the model input (test mode requires batch size set to 1)
    dummy_input = torch.randn((1, *input_shape))
    quantizer = torch_quantizer("test", model, (dummy_input), output_dir=str(output_dir))
    quant_model = quantizer.quant_model

    # NOTICE: Although calculating metrics on the quantized model is not necessary, you have to run at least one
    # sample in the "test" mode before exporting the xmodel.
    # Test the quantized model, save the predictions to a h5 file to calculate the metics on the host machine
    with h5py.File(data_h5_path, "r") as f_in, h5py.File(test_samples, "w") as f_out:
        sample_names = list(f_in["test"].keys())[:samples_num_limit]
        for sample_name in tqdm(sample_names):
            input_image = torch.as_tensor(f_in[f"test/{sample_name}"][:])
            input_batch = input_image.unsqueeze(0)
            pred = quant_model(input_batch)
            f_out.create_dataset(sample_name, data=pred.detach())

    quantizer.export_xmodel(str(output_dir))


def main(
    input_size: List[int],
    num_classes: int,
    calib_batch_size: int,
    calib_samples_limit: Optional[int],
    test_samples_limit: Optional[int],
    quantization_samples: Path,
    test_samples: Path,
    state_dict: Path,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the model
    model = Unet(num_classes=num_classes)
    model.load_state_dict(torch.load(state_dict))

    # Perform calibration based on train samples
    calibrate(model, input_size, calib_batch_size, quantization_samples, output_dir, calib_samples_limit)

    # Test the quantized model and export the xmodel
    test(model, input_size, quantization_samples, output_dir, test_samples, test_samples_limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-size", type=int, nargs="+", default=DEFAULT_MODEL_INPUT_SIZE)
    parser.add_argument("--num-classes", type=int, default=DEFAULT_MODEL_NUM_CLASSES)
    parser.add_argument("--calib-batch-size", type=int, default=8)
    parser.add_argument("--calib-samples-limit", type=int, default=None)
    parser.add_argument("--test-samples-limit", type=int, default=None)
    parser.add_argument("--quantization-samples", type=Path, required=True)
    parser.add_argument("--test-samples", type=Path, required=True)
    parser.add_argument("--state-dict", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    main(**vars(args))
