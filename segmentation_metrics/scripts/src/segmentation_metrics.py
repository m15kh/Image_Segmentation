import os
import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.spatial.distance import directed_hausdorff


class SegmentationEvaluator:
    def __init__(self, gt_dir, pred_dir, save_dir=None):
        """
        Initialize the SegmentationEvaluator with directories.
        
        Args:
            gt_dir: Directory containing ground truth masks
            pred_dir: Directory containing predicted masks
            save_dir: Directory to save results (optional)
        """
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir
        self.save_dir = save_dir
    
    def dice_coefficient(self, y_true, y_pred):
        """
        Calculate the Dice coefficient between two binary masks.
        
        Args:
            y_true: Ground truth binary mask
            y_pred: Predicted binary mask
            
        Returns:
            dice_coef: Dice coefficient value
        """
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        
        intersection = np.sum(y_true_f * y_pred_f)
        smooth = 1e-6  # to avoid division by zero
        
        return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

    def iou_score(self, y_true, y_pred):
        """
        Calculate the Intersection over Union (IoU) between two binary masks.
        
        Args:
            y_true: Ground truth binary mask
            y_pred: Predicted binary mask
            
        Returns:
            iou: IoU score value
        """
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        
        intersection = np.sum(y_true_f * y_pred_f)
        union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
        smooth = 1e-6  # to avoid division by zero
        
        return (intersection + smooth) / (union + smooth)

    def precision_score(self, y_true, y_pred):
        """
        Calculate the precision (positive predictive value) between two binary masks.
        
        Args:
            y_true: Ground truth binary mask
            y_pred: Predicted binary mask
            
        Returns:
            precision: Precision score value
        """
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        
        true_positive = np.sum(y_true_f * y_pred_f)
        total_predicted_positive = np.sum(y_pred_f)
        smooth = 1e-6  # to avoid division by zero
        
        return (true_positive + smooth) / (total_predicted_positive + smooth)

    def recall_score(self, y_true, y_pred):
        """
        Calculate the recall (sensitivity) between two binary masks.
        
        Args:
            y_true: Ground truth binary mask
            y_pred: Predicted binary mask
            
        Returns:
            recall: Recall score value
        """
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        
        true_positive = np.sum(y_true_f * y_pred_f)
        total_actual_positive = np.sum(y_true_f)
        smooth = 1e-6  # to avoid division by zero
        
        return (true_positive + smooth) / (total_actual_positive + smooth)

    def pixel_accuracy(self, y_true, y_pred):
        """
        Calculate pixel accuracy between two binary masks.
        
        Args:
            y_true: Ground truth binary mask
            y_pred: Predicted binary mask
            
        Returns:
            accuracy: Pixel accuracy value
        """
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        
        correct_pixels = np.sum(y_true_f == y_pred_f)
        total_pixels = len(y_true_f)
        
        return correct_pixels / total_pixels
    
    def compute_roc_auc(self, y_true, y_pred):
        """
        Calculate ROC-AUC score between two binary masks.
        
        Args:
            y_true: Ground truth binary mask
            y_pred: Predicted binary mask
            
        Returns:
            auc: ROC-AUC score value
        """
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        
        try:
            return roc_auc_score(y_true_f, y_pred_f)
        except ValueError:
            # This can happen if all predictions are the same class
            return np.nan
    
    def contour_accuracy(self, y_true, y_pred):
        """
        Calculate contour accuracy between two binary masks.
        
        Args:
            y_true: Ground truth binary mask
            y_pred: Predicted binary mask
            
        Returns:
            accuracy: Contour accuracy value
        """
        # Extract contours from ground truth and prediction
        contours_true, _ = cv2.findContours(y_true, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_pred, _ = cv2.findContours(y_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create empty masks with only contours
        h, w = y_true.shape
        mask_true = np.zeros((h, w), dtype=np.uint8)
        mask_pred = np.zeros((h, w), dtype=np.uint8)
        
        cv2.drawContours(mask_true, contours_true, -1, 1, 1)
        cv2.drawContours(mask_pred, contours_pred, -1, 1, 1)
        
        # Calculate overlap between contours
        intersection = np.sum(mask_true * mask_pred)
        union = np.sum(mask_true) + np.sum(mask_pred) - intersection
        
        if union == 0:
            return 1.0  # If both masks have no contours, consider it perfect match
        
        return intersection / union
    
    def compute_hausdorff_distance(self, y_true, y_pred):
        """
        Calculate the Hausdorff distance between two binary masks.
        
        Args:
            y_true: Ground truth binary mask
            y_pred: Predicted binary mask
            
        Returns:
            distance: Hausdorff distance value
        """
        # Get coordinates of boundary pixels
        y_true_points = np.array(np.where(y_true > 0)).T
        y_pred_points = np.array(np.where(y_pred > 0)).T
        
        if len(y_true_points) == 0 or len(y_pred_points) == 0:
            return np.nan  # Return NaN if either mask is empty
        
        # Calculate directed Hausdorff distances
        forward_hausdorff, _, _ = directed_hausdorff(y_true_points, y_pred_points)
        reverse_hausdorff, _, _ = directed_hausdorff(y_pred_points, y_true_points)
        
        # Return the maximum of the two directed distances
        return max(forward_hausdorff, reverse_hausdorff)

    def load_and_process_images(self, gt_file, pred_file, resize_target='pred'):
        """
        Load and preprocess ground truth and prediction images.

        Args:
            gt_file: Filename of ground truth mask
            pred_file: Filename of predicted mask
            resize_target: Target to resize if dimensions mismatch ('gt' or 'pred')

        Returns:
            gt_mask: Processed ground truth mask
            pred_mask: Processed prediction mask
            success: Boolean indicating if loading was successful
        """
        # Check if prediction file exists
        if not os.path.exists(os.path.join(self.pred_dir, pred_file)):
            print(f"Warning: No corresponding prediction found for {gt_file}")
            return None, None, False

        # Read ground truth and prediction masks
        gt_mask = cv2.imread(os.path.join(self.gt_dir, gt_file), cv2.IMREAD_GRAYSCALE)
        pred_mask = cv2.imread(os.path.join(self.pred_dir, pred_file), cv2.IMREAD_GRAYSCALE)

        # Ensure masks have the same dimensions
        if gt_mask.shape != pred_mask.shape:
            print(f"Warning: Shape mismatch for {gt_file}. Resizing {resize_target}.")
            if resize_target == 'gt':
                gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]))
            elif resize_target == 'pred':
                pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))
            else:
                print(f"Error: Invalid resize_target '{resize_target}'. Must be 'gt' or 'pred'.")
                return None, None, False

        # Binarize masks after resizing
        gt_mask = (gt_mask > 0).astype(np.uint8)
        pred_mask = (pred_mask > 0).astype(np.uint8)

        return gt_mask, pred_mask, True

    def evaluate(self):
        """
        Evaluate segmentation performance using Dice, IoU, Precision, Recall,
        Pixel Accuracy, ROC-AUC, Contour Accuracy, and Hausdorff Distance.

        Returns:
            mean_metrics: Dictionary containing mean values of all metrics
            all_metrics: Dictionary containing lists of all metrics
        """
        # Get all files in the ground truth directory
        gt_files = sorted(os.listdir(self.gt_dir))

        # Load resize_target from config
        resize_target = config.get('metrics', {}).get('resize_target', 'pred')

        # Initialize lists to store metrics
        all_dice = []
        all_iou = []
        all_precision = []
        all_recall = []
        all_pixel_accuracy = []
        all_roc_auc = []
        all_contour_accuracy = []
        all_hausdorff_distance = []
        image_names = []

        # Loop through each ground truth file
        for gt_file in gt_files:
            # Find corresponding prediction file
            pred_file = gt_file  # Assuming same filename for ground truth and prediction

            # Load and process images
            gt_mask, pred_mask, success = self.load_and_process_images(gt_file, pred_file, resize_target)
            if not success:
                continue

            # Calculate metrics
            dice = self.dice_coefficient(gt_mask, pred_mask)
            iou = self.iou_score(gt_mask, pred_mask)
            precision = self.precision_score(gt_mask, pred_mask)
            recall = self.recall_score(gt_mask, pred_mask)
            accuracy = self.pixel_accuracy(gt_mask, pred_mask)
            roc_auc = self.compute_roc_auc(gt_mask, pred_mask)
            contour_acc = self.contour_accuracy(gt_mask, pred_mask)
            hausdorff_dist = self.compute_hausdorff_distance(gt_mask, pred_mask)

            all_dice.append(dice)
            all_iou.append(iou)
            all_precision.append(precision)
            all_recall.append(recall)
            all_pixel_accuracy.append(accuracy)
            all_roc_auc.append(roc_auc)
            all_contour_accuracy.append(contour_acc)
            all_hausdorff_distance.append(hausdorff_dist)
            image_names.append(gt_file)

            print(f"{gt_file}: Dice={dice:.4f}, IoU={iou:.4f}, Precision={precision:.4f}, "
                  f"Recall={recall:.4f}, Accuracy={accuracy:.4f}, ROC-AUC={roc_auc:.4f}, "
                  f"Contour={contour_acc:.4f}, Hausdorff={hausdorff_dist:.4f}")

        # Calculate mean metrics
        mean_dice = np.nanmean(all_dice)
        mean_iou = np.nanmean(all_iou)
        mean_precision = np.nanmean(all_precision)
        mean_recall = np.nanmean(all_recall)
        mean_pixel_accuracy = np.nanmean(all_pixel_accuracy)
        mean_roc_auc = np.nanmean(all_roc_auc)
        mean_contour_accuracy = np.nanmean(all_contour_accuracy)
        mean_hausdorff_distance = np.nanmean(all_hausdorff_distance)

        print(f"Mean Dice: {mean_dice * 100:.2f}%")
        print(f"Mean IoU: {mean_iou * 100:.2f}%")
        print(f"Mean Precision: {mean_precision * 100:.2f}%")
        print(f"Mean Recall: {mean_recall * 100:.2f}%")
        print(f"Mean Pixel Accuracy: {mean_pixel_accuracy * 100:.2f}%")
        print(f"Mean ROC-AUC: {mean_roc_auc:.4f}")
        print(f"Mean Contour Accuracy: {mean_contour_accuracy:.4f}")
        print(f"Mean Hausdorff Distance: {mean_hausdorff_distance:.4f}")

        # Save results if save_dir is provided
        if self.save_dir:
            mean_metrics = {
                'dice': mean_dice,
                'iou': mean_iou,
                'precision': mean_precision,
                'recall': mean_recall,
                'pixel_accuracy': mean_pixel_accuracy,
                'roc_auc': mean_roc_auc,
                'contour_accuracy': mean_contour_accuracy,
                'hausdorff_distance': mean_hausdorff_distance
            }
            
            all_metrics = {
                'dice': all_dice,
                'iou': all_iou,
                'precision': all_precision,
                'recall': all_recall,
                'pixel_accuracy': all_pixel_accuracy,
                'roc_auc': all_roc_auc,
                'contour_accuracy': all_contour_accuracy,
                'hausdorff_distance': all_hausdorff_distance
            }
            
            self.save_results(mean_metrics, all_metrics, image_names)
            
            return mean_metrics, all_metrics
        
        all_metrics = {
            'dice': all_dice,
            'iou': all_iou,
            'precision': all_precision,
            'recall': all_recall,
            'pixel_accuracy': all_pixel_accuracy,
            'roc_auc': all_roc_auc,
            'contour_accuracy': all_contour_accuracy,
            'hausdorff_distance': all_hausdorff_distance
        }
        
        mean_metrics = {
            'dice': mean_dice,
            'iou': mean_iou,
            'precision': mean_precision,
            'recall': mean_recall,
            'pixel_accuracy': mean_pixel_accuracy,
            'roc_auc': mean_roc_auc,
            'contour_accuracy': mean_contour_accuracy,
            'hausdorff_distance': mean_hausdorff_distance
        }

        return mean_metrics, all_metrics

    def save_results(self, mean_metrics, all_metrics, image_names):
        """
        Save evaluation results as text file and plots.
        
        Args:
            mean_metrics: Dictionary of mean metric values
            all_metrics: Dictionary containing lists of all metrics
            image_names: List of image filenames
        """
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Save results as text file
        with open(os.path.join(self.save_dir, 'segmentation_metrics.txt'), 'w') as f:
            f.write(f"Mean Dice coefficient: {mean_metrics['dice']:.4f}\n")
            f.write(f"Mean IoU: {mean_metrics['iou']:.4f}\n")
            f.write(f"Mean Precision: {mean_metrics['precision']:.4f}\n")
            f.write(f"Mean Recall: {mean_metrics['recall']:.4f}\n")
            f.write(f"Mean Pixel Accuracy: {mean_metrics['pixel_accuracy']:.4f}\n")
            f.write(f"Mean ROC-AUC: {mean_metrics['roc_auc']:.4f}\n")
            f.write(f"Mean Contour Accuracy: {mean_metrics['contour_accuracy']:.4f}\n")
            f.write(f"Mean Hausdorff Distance: {mean_metrics['hausdorff_distance']:.4f}\n\n")
            f.write("Individual results:\n")
            
            # Create a table with image names and scores
            table_data = []
            for i, name in enumerate(image_names):
                row = [i+1, name]
                for metric_name in ['dice', 'iou', 'precision', 'recall', 'pixel_accuracy', 
                                    'roc_auc', 'contour_accuracy', 'hausdorff_distance']:
                    row.append(f"{all_metrics[metric_name][i]:.4f}")
                table_data.append(row)
                
            headers = ["#", "Image", "Dice", "IoU", "Precision", "Recall", "Pixel Accuracy", 
                       "ROC-AUC", "Contour Acc", "Hausdorff Dist"]
            f.write(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Create plots for each metric
        metrics_plots = {
            'dice_plot.png': ('Dice Coefficient', all_metrics['dice'], mean_metrics['dice'], 'skyblue'),
            'iou_plot.png': ('IoU Score', all_metrics['iou'], mean_metrics['iou'], 'lightgreen'),
            'precision_plot.png': ('Precision', all_metrics['precision'], mean_metrics['precision'], 'salmon'),
            'recall_plot.png': ('Recall', all_metrics['recall'], mean_metrics['recall'], 'lightblue'),
            'pixel_accuracy_plot.png': ('Pixel Accuracy', all_metrics['pixel_accuracy'], mean_metrics['pixel_accuracy'], 'mediumpurple'),
            'roc_auc_plot.png': ('ROC-AUC', all_metrics['roc_auc'], mean_metrics['roc_auc'], 'lightcoral'),
            'contour_accuracy_plot.png': ('Contour Accuracy', all_metrics['contour_accuracy'], mean_metrics['contour_accuracy'], 'khaki'),
            'hausdorff_distance_plot.png': ('Hausdorff Distance', all_metrics['hausdorff_distance'], mean_metrics['hausdorff_distance'], 'lightseagreen')
        }
        
        for filename, (title, values, mean_val, color) in metrics_plots.items():
            plt.figure(figsize=(12, 6))
            plt.bar(image_names, values, color=color)
            plt.axhline(y=mean_val, color='r', linestyle='-', label=f'Mean {title}: {mean_val:.4f}')
            plt.xlabel('Image')
            plt.ylabel(title)
            plt.title(f'{title} per Image')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.legend()
            plt.savefig(os.path.join(self.save_dir, filename))


if __name__ == "__main__":
    # Load configuration from YAML file
    config_path = '/home/ubuntu/m15kh/own/segmentation_metrics/config.yaml'
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}")
        print("Creating default config file...")
        
        # Create default config
        default_config = {
            'segmentation': {
                'gt_dir': '/home/ubuntu/m15kh/Image_Segmentation/Validation/masks/',
                'pred_dir': '/home/ubuntu/m15kh/Image_Segmentation/FineTune_DeepLabV3/outputs_validation/inference_results_validation/',
                'save_dir': '/home/ubuntu/m15kh/Image_Segmentation/FineTune_DeepLabV3/metrics_result/'
            }
        }
        
        # Save default config
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
            
        print(f"Default config file created at {config_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get segmentation config
    segmentation_config = config.get('metrics', {})
    gt_dir = segmentation_config.get('gt_dir', '')
    pred_dir = segmentation_config.get('pred_dir', '')
    save_dir = segmentation_config.get('save_dir', '')
    
    print(f"Using configuration:")
    print(f"  GT Directory: {gt_dir}")
    print(f"  Prediction Directory: {pred_dir}")
    print(f"  Save Directory: {save_dir}")
    
    # Create evaluator and run evaluation
    evaluator = SegmentationEvaluator(gt_dir, pred_dir, save_dir)
    evaluator.evaluate()
