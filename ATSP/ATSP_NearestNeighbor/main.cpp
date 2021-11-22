
/*
The MIT License

Copyright (c) 2021 MatNet

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DATA_DIR "../data/n20/"
#define FILENAME_PREFIX "problem_20_0_1000000_"

#define PRINT_RESULT_TRAJ false
#define NUM_DATA 10000

#define MAX_LEN 100
#define MAX_LINE_BUFF 2048


unsigned int N;
unsigned int batch_dist[NUM_DATA][MAX_LEN][MAX_LEN];

bool load_data_file(char* filepath, int data_index)
{
	unsigned int (*dist)[MAX_LEN] = batch_dist[data_index];
	
	FILE* fp = fopen(filepath, "r");
	char buff[MAX_LINE_BUFF];
	
	if ( fp==NULL ) 
	{
		printf("Error: Can't open %s", filepath);
		return false;
	}

	fgets(buff, MAX_LINE_BUFF, fp);
	fgets(buff, MAX_LINE_BUFF, fp);

	N = atoi(buff+11);

	fgets(buff, MAX_LINE_BUFF, fp);
	fgets(buff, MAX_LINE_BUFF, fp);
	fgets(buff, MAX_LINE_BUFF, fp);

	for ( int i = 0 ; i < N; i ++)		
		for(int j = 0 ; j < N; j ++) 
			fscanf(fp, "%d", &dist[i][j]);
	
	fclose(fp);
	
	return true;
}

void print_data()
{
	unsigned int (*dist)[MAX_LEN] = batch_dist[0];

	printf("N=%d\n", N);
	
	for ( int i = 0 ; i < N ; i ++)
	{
		for ( int j = 0 ; j < N ; j ++) printf("%d\t", dist[i][j]);			
		printf("\n");
	}
}

unsigned int atsp_nn(int data_index)
{
	unsigned int (*dist)[MAX_LEN] = batch_dist[data_index];

	bool visited[MAX_LEN];
	int position[MAX_LEN];
	unsigned int total_len = 0;
	
	for (int i = 0 ; i < N; i ++) visited[i] = false;

	int curr_pos = 0;
	visited[0] = true;

	#if PRINT_RESULT_TRAJ==true
	printf("%d ", 0);
	#endif
	
	for( int i = 1; i < N ; i++ )
	{		
		int next_pos = curr_pos;
			
		for( int j = 0 ; j < N; j++)
		{
			if (visited[j]) continue;
			if (dist[curr_pos][j] < dist[curr_pos][next_pos]) next_pos = j;
		}

		total_len += dist[curr_pos][next_pos];
		curr_pos = next_pos;
		visited[curr_pos] = true;

		#if PRINT_RESULT_TRAJ==true
		printf("%d ", curr_pos);
		#endif
	}

	#if PRINT_RESULT_TRAJ==true
	printf("\n", curr_pos);
	#endif

	total_len += dist[curr_pos][0];

	return total_len;
}


int main()
{
	clock_t start, end;
	unsigned long total_dist=0;

	char dir[1024] = DATA_DIR;
	char fn_prefix[1024] = FILENAME_PREFIX;	
	char fn_postfix[1024] = ".atsp";

	printf("============================================\n");
	printf("dir: %s\n", dir);
	printf("filename_prefix: %s\n", fn_prefix);

	printf("start loading data\n");
	for ( int i = 0 ; i < NUM_DATA; i ++) 
	{
		char filename[2048];
		sprintf(filename, "%s%s%d%s", dir, fn_prefix, i, fn_postfix);
		printf("%cloading: %s", 13, filename);
		load_data_file(filename, i);
	}
	printf("\ncomplete loading data\n");

	start = clock();
	for ( int i = 0 ; i < NUM_DATA; i ++) total_dist += (unsigned long)atsp_nn(i);
	end = clock();
		
	printf("total_dist = %.2lf, time=%d ms\n", (double)total_dist/(double)NUM_DATA, (end-start)/(CLOCKS_PER_SEC/1000));
	printf("============================================\n");
}
