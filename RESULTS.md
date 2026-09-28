# Experimental Results

This page presents the experimental results reported in `results.docx`.\
The raw five-run results and statistical summary are also available in
`validation_results.csv` and `validation_summary.csv`.

## Dataset and Evaluation Setup

The evaluation loaded **405,894 records**. The generated edge-cloud
labels contained **317,286 Cloud** samples and **88,608 Edge** samples.

The model was trained for **35 epochs**. The reported training time was
**221.9726 seconds**.

## Classification Performance

  Metric                                     Result
  ----------------------------- -------------------
  Accuracy                                 98.2138%
  Precision                                92.8541%
  Recall                                   99.4453%
  F1 Score                                 96.0367%
  Average Scheduling Overhead     0.000977 sec/task

The classification report reported Cloud precision/recall/F1 of **1.00 /
0.98 / 0.99** and Edge precision/recall/F1 of **0.93 / 0.99 / 0.96**.

## Scalability Results

    Number of Tasks   Average Latency   Success Rate
  ----------------- ----------------- --------------
              1,000          0.003147         48.94%
              5,000          0.003055         48.94%
             10,000          0.002994         48.94%
             20,000          0.003024         48.94%

## Load Balancing Results

  Metric                                       Result
  ------------------------------ --------------------
  Memory Load Variance                    151264.1263
  Normalized CPU Load Variance                 0.0002
  Balancing Accuracy               79.08142719072166%
  Hotspots Detected                                 0

## Task Performance Metrics

  Metric                         Result
  ---------------------- --------------
  Average Task Latency     0.0181 units
  Deadline Miss Rate              0.00%
  Average Waiting Time     0.0126 units

## Edge-Cloud Orchestration

  Metric                                                    Result
  ------------------------------------------- --------------------
  Edge/Cloud Offloading Accuracy                98.21382377215782%
  Average Prediction Time per Task                     0.157383 ms
  Baseline Cloud Offloads                                   63,513
  Predicted Cloud Offloads                                  62,259
  Reduction in Unnecessary Cloud Offloading                  1,254
  Reduction Percentage                                       1.97%

## Fault Tolerance and Reliability

  Metric                         Result
  --------------------------- ---------
  Total Tasks                   405,894
  Successfully Assigned         198,656
  Failed Assignments            207,238
  Average Failover Attempts        2.02
  Total Critical Tasks          333,885
  Failed Critical Tasks         169,611

## Additional Metrics

  Metric                               Result
  --------------------------------- ---------
  Average Latency Reduction            20.14%
  Task Completion Rate                 48.94%
  Load Variance (CPU)                  0.0002
  Task Migration Success Rate         100.00%
  Orchestration Decision Accuracy      48.11%

## Result Figures

The following figures were extracted directly from the original
`results.docx`. They are stored in `results_images/` so GitHub can
render them directly on this page.

### Figure 1

![Result Figure 1](results_images/image1.png)

### Figure 2

![Result Figure 2](results_images/image2.png)

### Figure 3

![Result Figure 3](results_images/image3.png)

### Figure 4

![Result Figure 4](results_images/image4.png)

### Figure 5

![Result Figure 5](results_images/image5.png)

### Figure 6

![Result Figure 6](results_images/image6.png)

### Figure 7

![Result Figure 7](results_images/image7.png)

### Figure 8

![Result Figure 8](results_images/image8.png)

### Figure 9

![Result Figure 9](results_images/image9.png)

### Figure 10

![Result Figure 10](results_images/image10.png)

### Figure 11

![Result Figure 11](results_images/image11.png)

### Figure 12

![Result Figure 12](results_images/image12.png)

### Figure 13

![Result Figure 13](results_images/image13.png)

### Figure 14

![Result Figure 14](results_images/image14.png)

### Figure 15

![Result Figure 15](results_images/image15.png)

### Figure 16

![Result Figure 16](results_images/image16.png)

### Figure 17

![Result Figure 17](results_images/image17.png)

### Figure 18

![Result Figure 18](results_images/image18.png)

### Figure 19

![Result Figure 19](results_images/image19.png)

### Figure 20

![Result Figure 20](results_images/image20.png)

### Figure 21

![Result Figure 21](results_images/image21.png)

## Reproducibility Files

For numerical reproducibility, refer to:

-   `validation_results.csv` --- raw results from the five experimental
    runs.
-   `validation_summary.csv` --- statistical summary of the five runs.
-   `statistical_validation.py` --- statistical validation procedure.
-   `main.py` --- main implementation and preprocessing applied to the
    released input CSV.
-   `REPRODUCIBILITY.md` --- configuration, execution procedure, and
    reproducibility scope.
