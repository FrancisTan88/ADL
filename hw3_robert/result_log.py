import json
from subprocess import check_output, CalledProcessError, STDOUT, run
from pathlib import Path
from matplotlib import pyplot as plt
from collections import defaultdict


def save_curve_plot(log):
    plt_name = 'Learning_Curves'
    for eval_type, values in log.items():
        x = list(range(1, len(values) + 1))
        # plotting the line 2 points
        plt.plot(x, values, label=eval_type)

    # naming the x axis
    plt.xlabel('training steps (500 batch per step)')
    plt.title(plt_name)
    # show a legend on the plot
    plt.legend()
    plt.savefig(f'{plt_name}.png')
    plt.close()


def save_learning_curve():
    test_path = Path('data/public.jsonl')
    base_dir = Path('ckpt')
    base_name = 'checkpoint-'
    steps = sorted([int(str(filename).split('-')[-1])
                    for filename in base_dir.iterdir() if filename.name.startswith(base_name)])
    log = defaultdict(list)
    for step in steps:
        predict_path = Path('predictions') / f'{base_name}{step}.jsonl'
        predict_script = ['python', 'run_summarization_no_trainer.py', '--model_name_or_path',
                          base_dir / f'{base_name}{step}', '--text_column', 'maintext',
                          '--validation_file', str(test_path), '--output_path', predict_path,
                          '--num_beams', '8']
        run(predict_script)
        eval_script = ['python', 'eval.py', '-r', str(test_path), '-s', str(predict_path)]
        result = check_output(eval_script)
        if isinstance(result, bytes):
            result = result.decode('utf-8')
        result = json.loads(result)
        for key, value in result.items():
            log[key].append(value.get('f'))
    save_curve_plot(log)


def print_ablation():
    base_dir = Path('predictions')
    for filename in base_dir.iterdir():
        script = ['python', 'eval.py', '-r', 'data/public.jsonl', '-s', filename]
        output = check_output(script)
        if isinstance(output, bytes):
            output = output.decode('utf-8')
        print()
        print('=' * 50)
        print(filename)
        print(output)
        print('=' * 50)


if __name__ == '__main__':
    save_learning_curve()
