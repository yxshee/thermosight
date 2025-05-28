# Troubleshooting

Common issues and solutions for ThermoSight.

---

## 1. CUDA Out of Memory

- Reduce `batch_size` in training scripts.
- Ensure no other GPU processes are running.

## 2. Data Not Found

- Check that your images are in `data/raw/` and organized by class.
- Verify paths in your scripts.

## 3. Model Not Improving

- Try lowering the learning rate.
- Increase the number of epochs.
- Check your data for class imbalance.

## 4. TensorBoard Not Showing Logs

- Make sure you are pointing to the correct log directory: `outputs/logs`.

## 5. Import Errors

- Ensure you are running scripts from the project root.
- Use `python -m` if necessary, e.g., `python -m src.models.train`.

---
