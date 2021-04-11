import click

from config import INITIAL_PARAM, PARAM_1, PARAM_2, WIN_SIZE, TOP, RANK 
from utils import load_data, plot_corr, evaluate
from model import train_xgbr, make_prediction


@click.command()
@click.option('train_file', '--train', type=click.Path(exists=True), 
              help="Training data file path; if provided, retrains the model and overwrite to model_path")
@click.option('test_file', '--test', default='./data/sample/test.csv', type=click.Path(exists=True), 
              help="Testing data file path; if not provided sample data at ./data/sample/test.csv is used")
@click.option('model_path', '--model', default='./model', type=click.Path(exists=True), 
              help='Directory path to model; default path ./model')
@click.option('output_path', '--output', default='./output', type=click.Path(exists=True),
             help='Direcotry path to pred.csv; default path./output')
@click.option('plot_flag', '--plot', is_flag=True, 
              help="Flag if or not plotting the correlation heatmap, only valid if training data is provided")
def main(train_file, test_file, model_path, output_path, plot_flag):
              
    if train_file:
        click.echo('Loading training data')
        X_train, y_train = load_data(train_file, model_path) 
        
        click.echo('Generating corr heatmap')
        if plot_flag:
            try:
                plot_corr(X_train, y_train, output_path, plot_it=False)
            except:
                click.echo('Failed to plot correlation heatmap')
        
        click.echo('Model traninig')
        train_xgbr(X_train, y_train, INITIAL_PARAM, PARAM_1, PARAM_2, model_path=model_path, test_size=0.2)
    
    click.echo('Loading test data')
    X_test, y_test = load_data(test_file, model_path)
    
    click.echo('Making prediction')
    pred = make_prediction(X_test, model_path, output_path)
    
    click.echo('Evaluating and plot')          
    _, scores = evaluate(pred, y_test, X_test, output_path, window_size=WIN_SIZE, top=TOP, ranks=RANK)
               

if __name__ == '__main__':
    main()