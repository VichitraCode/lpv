#include <bits/stdc++.h>
#include <omp.h>

using namespace std;

int N = 6, a[100];

void max_red(int a[], int n) {
	int mx = INT_MIN;
	
	#pragma omp parallel for reduction(max: mx)
	for(int i = 0; i < n; i++) if(a[i] > mx) mx = a[i];
	
	cout << "Maximum value: " << mx << endl; 
}
void min_red(int a[], int n) {
	int mn = INT_MAX;
	
	#pragma omp parallel for reduction(min: mn)
	for(int i = 0; i < n; i++) if(a[i] < mn) mn = a[i];
	
	cout << "Minimum value: " << mn << endl; 
}


void sum_red(int a[], int n) {
	int sum = 0;
	
	#pragma omp parallel for reduction(+: sum)
	for(int i = 0; i < n; i++) sum += a[i];
	
	cout << "Sum: " << sum << endl; 
}


void avg_red(int a[], int n) {
	double sum = 0, cnt = n;
	
	#pragma omp parallel for reduction(+: sum)
	for(int i = 0; i < n; i++) sum += a[i];
	double avg = sum / cnt;
	cout << "Average: " << avg << endl; 
}

int main() {
	cout << "Enter " << N << " elements: ";
	int a[N] ;	
	for(int i = 0; i < N; i++) cin >> a[i];
	
	max_red(a, N);
	min_red(a, N);
	sum_red(a, N);
	avg_red(a, N);
	max_red(a, N);
}

/* 
Sample Input :
3 1 8 0 5

*/