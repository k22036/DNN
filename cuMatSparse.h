/*
 * cuMatSparse.h
 *
 *  Created on: 2016/02/24
 *      Author: takeshi.fujita
 */

#ifndef CUMATSPARSE_H_
#define CUMATSPARSE_H_

#include<iostream>
#include<cuda_runtime_api.h>
#include<cublas_v2.h>
#include<cusparse_v2.h>
#include<thrust/device_vector.h>

#include "cuMat.h"

class cuMatSparse {

public:

    int rows = 0;
    int cols = 0;

    cusparseHandle_t cuHandle;
    cusparseMatDescr_t descr;

    float *csrVal = NULL;
    int *csrRowPtr = NULL;
    int *csrColInd = NULL;

    float *csrValDevice = NULL;
    int *csrRowPtrDevice = NULL;
    int *csrColIndDevice = NULL;


    int numVals = 0;

    cuMat rt, bt;


    cuMatSparse(){
        cusparseCreate(&cuHandle);
        cusparseCreateMatDescr(&descr);
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    }

    cuMatSparse(int rows, int cols, int numberOfVals){
        cout << "cuMatSparse(int rows, int numberOfVals)" << endl;
        cusparseCreate(&cuHandle);
        cusparseCreateMatDescr(&descr);
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

        new_matrix(rows, cols, numberOfVals);
    }

    cuMatSparse(vector<float> &ids, int col_nums) : cuMatSparse(){
        embed(ids, col_nums);
    }


    ~cuMatSparse(){
        cusparseDestroyMatDescr(descr);
        cusparseDestroy(cuHandle);

        free(csrVal);
        free(csrRowPtr);
        free(csrColInd);
        cudaFree(csrValDevice);
        cudaFree(csrRowPtrDevice);
        cudaFree(csrColIndDevice);


    }

    void new_matrix(int rows, int cols, int numberOfVals){
        this->rows = rows;
        this->cols = cols;
        this->numVals = numberOfVals;

        cudaError_t error = cudaMalloc((void**) &csrValDevice, numberOfVals * sizeof(*csrValDevice));
        error = cudaMalloc((void**) &csrRowPtrDevice, (rows+1) * sizeof(*csrRowPtrDevice));
        error = cudaMalloc((void**) &csrColIndDevice, numberOfVals * sizeof(*csrColIndDevice));

        cudaMemset(csrValDevice, 0x00, numberOfVals * sizeof(*csrValDevice));
        cudaMemset(csrRowPtrDevice, 0x00, (rows+1)  * sizeof(*csrRowPtrDevice));
        cudaMemset(csrColIndDevice, 0x00, numberOfVals * sizeof(*csrColIndDevice));
    }

    cuMatSparse &operator=(const cuMatSparse &a) {

        new_matrix(a.rows, a.cols, a.numVals);

        cudaError_t error = cudaMemcpy(csrValDevice, a.csrValDevice, a.numVals * sizeof(*csrValDevice), cudaMemcpyDeviceToDevice);
        error = cudaMemcpy(csrRowPtrDevice, a.csrRowPtrDevice, (a.rows+1) * sizeof(*csrRowPtrDevice), cudaMemcpyDeviceToDevice);
        error = cudaMemcpy(csrColIndDevice, a.csrColIndDevice, a.numVals * sizeof(*csrColIndDevice), cudaMemcpyDeviceToDevice);

        //cout << "a.rows:" << a.rows << " a.cols:" << a.cols << endl;
        //cout << "this->rows:" << this->rows << " this->cols:" << this->cols << endl;

        return *this;
    }

    void zeros(){
        cudaMemset(csrValDevice, 0x00, numVals * sizeof(*csrValDevice));
        cudaMemset(csrRowPtrDevice, 0x00, (rows+1)  * sizeof(*csrRowPtrDevice));
        cudaMemset(csrColIndDevice, 0x00, numVals * sizeof(*csrColIndDevice));
    }

    void memSetHost(float *v, int *r, int *c) {
        cudaError_t error = cudaMemcpy(csrValDevice, v, numVals * sizeof(*csrValDevice), cudaMemcpyHostToDevice);
        if (error != cudaSuccess)  printf("memSetHost cudaMemcpy error: csrValDevice\n");
        error = cudaMemcpy(csrRowPtrDevice, r, (rows+1) * sizeof(*csrRowPtrDevice), cudaMemcpyHostToDevice);
        if (error != cudaSuccess)  printf("memSetHost cudaMemcpy error: csrRowPtrDevice\n");
        error = cudaMemcpy(csrColIndDevice, c, numVals * sizeof(*csrColIndDevice), cudaMemcpyHostToDevice);
        if (error != cudaSuccess)  printf("memSetHost cudaMemcpy error: csrColIndDevice\n");
    }

    //column majar format
    void embed(vector<float> &ids, int col_nums){

            rows = ids.size();
            cols = col_nums;

            int num_vals = rows;
            numVals = num_vals;


            csrVal = (float *)malloc(num_vals * sizeof(*csrVal));
            csrRowPtr = (int *)malloc((rows+1) * sizeof(*csrRowPtr));
            csrColInd = (int *)malloc(num_vals * sizeof(*csrColInd));

            cudaError_t error = cudaMalloc((void**) &csrValDevice, num_vals * sizeof(*csrValDevice));
            error = cudaMalloc((void**) &csrRowPtrDevice, (rows+1) * sizeof(*csrRowPtrDevice));
            error = cudaMalloc((void**) &csrColIndDevice, num_vals * sizeof(*csrColIndDevice));


            memset(csrRowPtr, 0x00, (rows+1) * sizeof(*csrRowPtr));
            csrRowPtr[0] = 0;
            for(int i=0; i<rows; i++){
                csrVal[i] = 1.; //value is 1
                csrColInd[i] = ids[i];
                csrRowPtr[i+1] = csrRowPtr[i] + 1; //only a element per row
            }



            /*
            cout << "csrVal:" << endl;
            for(int i=0; i<num_vals; i++){
                cout << csrVal[i] << " ";
            }
            cout << endl;
            cout << "csrRowPtr:" << endl;
            for(int i=0; i<row_nums+1; i++){
                cout << csrRowPtr[i] << " ";
            }
            cout << endl;
            cout << "csrColInd:" << endl;
            for(int i=0; i<num_vals; i++){
                cout << csrColInd[i] << " ";
            }
            cout << endl;
            */

            memSetHost(csrVal, csrRowPtr, csrColInd);
        }


    /*
    //row majar format
    void embed(vector<float> &ids, int row_nums){

        rows = row_nums;
        cols = ids.size();

        int num_vals = cols;
        numVals = num_vals;

        csrVal = (float *)malloc(num_vals * sizeof(*csrVal));
        csrRowPtr = (int *)malloc((row_nums+1) * sizeof(*csrRowPtr));
        csrColInd = (int *)malloc(num_vals * sizeof(*csrColInd));

        cudaError_t error = cudaMalloc((void**) &csrValDevice, num_vals * sizeof(*csrValDevice));
        error = cudaMalloc((void**) &csrRowPtrDevice, (row_nums+1) * sizeof(*csrRowPtrDevice));
        error = cudaMalloc((void**) &csrColIndDevice, num_vals * sizeof(*csrColIndDevice));


        memset(csrRowPtr, 0x00, (row_nums+1) * sizeof(*csrRowPtr));
        int row_ptr_cnt = 0;
        csrRowPtr[0] = row_ptr_cnt;
        for(int i=0; i<num_vals; i++){
            csrVal[i] = 1.;

            for(int j=0; j<row_nums; j++){
                if (ids[i] == j){
                    row_ptr_cnt++;
                    csrColInd[i] = i;
                    csrRowPtr[j+1] += row_ptr_cnt;
                }
            }

        }
        for(int j=1; j<row_nums; j++){
            if (csrRowPtr[j] == 0.){
                csrRowPtr[j] = csrRowPtr[j-1];
            }
        }


        cout << "csrVal:" << endl;
        for(int i=0; i<num_vals; i++){
            cout << csrVal[i] << " ";
        }
        cout << endl;
        cout << "csrRowPtr:" << endl;
        for(int i=0; i<row_nums+1; i++){
            cout << csrRowPtr[i] << " ";
        }
        cout << endl;
        cout << "csrColInd:" << endl;
        for(int i=0; i<num_vals; i++){
            cout << csrColInd[i] << " ";
        }
        cout << endl;


        memSetHost(csrVal, csrRowPtr, csrColInd);
    }
    */

    /*
    friend ostream &operator<<(ostream &output, cuMat &a) {

        for(int i=0; i<numVals; i++){
            output << csrVal[i];
            output << endl;
        }
    }*/


    void s_s_dot(cuMatSparse &b, cuMatSparse &c){

        // cusparseStatus_t status =
        //         cusparseScsrgeam2(cuHandle,
        //                 CUSPARSE_OPERATION_NON_TRANSPOSE,
        //                 CUSPARSE_OPERATION_NON_TRANSPOSE,
        //                          rows,
        //                          b.cols,
        //                          cols,
        //                          descr,
        //                          numVals,
        //                          csrValDevice,
        //                          csrRowPtrDevice,
        //                          csrColIndDevice,
        //                          b.descr,
        //                          b.numVals,
        //                          b.csrValDevice,
        //                          b.csrRowPtrDevice,
        //                          b.csrColIndDevice,
        //                          c.descr,
        //                          c.csrValDevice,
        //                          c.csrRowPtrDevice,
        //                          c.csrColIndDevice );

        float alpha = 1.0f;
        float beta = 1.0f;

        // バッファサイズ計算
        size_t bufferSize;
        cusparseScsrgeam2_bufferSizeExt(cuHandle, rows, b.cols, &alpha, descr, numVals,
                                        csrValDevice, csrRowPtrDevice, csrColIndDevice,
                                        &beta, b.descr, b.numVals, b.csrValDevice,
                                        b.csrRowPtrDevice, b.csrColIndDevice,
                                        c.descr, c.csrValDevice, c.csrRowPtrDevice,
                                        c.csrColIndDevice, &bufferSize);

        // バッファメモリ確保
        void *buffer;
        cudaMalloc(&buffer, bufferSize);

        // cusparseScsrgeam2呼び出し
        cusparseStatus_t status = cusparseScsrgeam2(
            cuHandle, rows, b.cols,
            &alpha, descr, numVals,
            csrValDevice, csrRowPtrDevice, csrColIndDevice,
            &beta, b.descr, b.numVals, b.csrValDevice,
            b.csrRowPtrDevice, b.csrColIndDevice,
            c.descr, c.csrValDevice,
            c.csrRowPtrDevice, c.csrColIndDevice, buffer);

        if (status != CUSPARSE_STATUS_SUCCESS) {
            std::cerr << "CUSPARSE error: " << status << std::endl;
        }

        // バッファメモリ解放
        cudaFree(buffer);

        cudaDeviceSynchronize();

    }

    void s_d_dot(cuMat &b, cuMat &c) {
        float alpha = 1.0f;
        float beta = 0.0f;

        int blockDim = 4; // 修正: BSRブロックサイズを適切に設定 (例: 4)
        int mb = rows / blockDim; // 修正: BSR行列のブロック行数
        int kb = cols / blockDim; // 修正: BSR行列のブロック列数

        cusparseStatus_t status = cusparseSbsrmm(
            cuHandle,
            CUSPARSE_DIRECTION_ROW,                // 修正: BSR行列のストレージ方向
            CUSPARSE_OPERATION_NON_TRANSPOSE,     // 修正: Aの操作
            CUSPARSE_OPERATION_NON_TRANSPOSE,     // 修正: Bの操作
            mb,                                   // BSR行列Aのブロック行数
            b.cols,                               // 行列Bの列数
            kb,                                   // BSR行列Aのブロック列数
            numVals,                              // 非ゼロ要素数
            &alpha,                               // スカラーalpha
            descr,                                // Aのディスクリプタ
            csrValDevice,                         // BSR行列Aの値
            csrRowPtrDevice,                      // BSR行列Aの行ポインタ
            csrColIndDevice,                      // BSR行列Aの列インデックス
            blockDim,                             // ブロックサイズ
            b.mDevice,                            // 行列Bの値
            b.rows,                               // 行列Bのリード数
            &beta,                                // スカラーbeta
            c.mDevice,                            // 出力行列C
            c.rows);                              // 行列Cのリード数

        if (status != CUSPARSE_STATUS_SUCCESS) {
            std::cout << "ERROR: cuMatSparse::s_d_dot cusparseSbsrmm failed" << std::endl;
            std::cout << "a rows (blocks): " << mb << ", cols (blocks): " << kb << std::endl;
            std::cout << "b rows: " << b.rows << ", cols: " << b.cols << std::endl;
            std::cout << "c rows: " << c.rows << ", cols: " << c.cols << std::endl;

            switch (status) {
                case CUSPARSE_STATUS_NOT_INITIALIZED:
                    std::cout << "CUSPARSE_STATUS_NOT_INITIALIZED" << std::endl;
                    break;
                case CUSPARSE_STATUS_ALLOC_FAILED:
                    std::cout << "CUSPARSE_STATUS_ALLOC_FAILED" << std::endl;
                    break;
                case CUSPARSE_STATUS_INVALID_VALUE:
                    std::cout << "CUSPARSE_STATUS_INVALID_VALUE" << std::endl;
                    break;
                case CUSPARSE_STATUS_ARCH_MISMATCH:
                    std::cout << "CUSPARSE_STATUS_ARCH_MISMATCH" << std::endl;
                    break;
                case CUSPARSE_STATUS_EXECUTION_FAILED:
                    std::cout << "CUSPARSE_STATUS_EXECUTION_FAILED" << std::endl;
                    break;
                case CUSPARSE_STATUS_INTERNAL_ERROR:
                    std::cout << "CUSPARSE_STATUS_INTERNAL_ERROR" << std::endl;
                    break;
                case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
                    std::cout << "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED" << std::endl;
                    break;
            }
        }

        cudaDeviceSynchronize();
    }




    void d_s_dot(cuMat &b, cuMat &r){
        cuMatSparse t = this->transpose(); //waste time here

        if (rt.rows == 0){
            rt = r.transpose();
        }
        if (bt.rows == 0){
            bt = b.transpose();
        }
        b.transpose(bt);

        t.s_d_dot(bt, rt);

        rt.transpose(r);
    }


    void transpose(cuMatSparse &r) {
        cusparseStatus_t status;

        // 必要なバッファサイズを格納する変数
        size_t bufferSize = 0;
        void* dBuffer = nullptr;

        // バッファサイズを取得
        status = cusparseCsr2cscEx2_bufferSize(
            cuHandle,
            rows,                      // 行列の行数
            cols,                      // 行列の列数
            numVals,                   // 非ゼロ要素の数
            csrValDevice,              // 入力CSR行列の値
            csrRowPtrDevice,           // 入力CSR行列の行ポインタ
            csrColIndDevice,           // 入力CSR行列の列インデックス
            r.csrValDevice,            // 出力CSC行列の値
            r.csrRowPtrDevice,         // 出力CSC行列の列ポインタ
            r.csrColIndDevice,         // 出力CSC行列の行インデックス
            CUDA_R_32F,                // 修正: データ型 (単精度浮動小数点)
            CUSPARSE_ACTION_NUMERIC,   // 数値データを転送
            CUSPARSE_INDEX_BASE_ZERO,  // インデックスの基準 (0)
            CUSPARSE_CSR2CSC_ALG1,     // アルゴリズム選択
            &bufferSize);              // 必要なバッファサイズ

        if (status != CUSPARSE_STATUS_SUCCESS) {
            std::cerr << "Failed to calculate buffer size for CSR2CSC: " << status << std::endl;
            return;
        }

        // 作業バッファをデバイスメモリに確保
        cudaMalloc(&dBuffer, bufferSize);

        // CSR形式をCSC形式 (転置) に変換
        status = cusparseCsr2cscEx2(
            cuHandle,
            rows,
            cols,
            numVals,
            csrValDevice,
            csrRowPtrDevice,
            csrColIndDevice,
            r.csrValDevice,
            r.csrRowPtrDevice,
            r.csrColIndDevice,
            CUDA_R_32F,                // 修正: データ型
            CUSPARSE_ACTION_NUMERIC,   // 数値データを転送
            CUSPARSE_INDEX_BASE_ZERO,
            CUSPARSE_CSR2CSC_ALG1,
            dBuffer);

        if (status != CUSPARSE_STATUS_SUCCESS) {
            std::cerr << "Failed to transpose CSR to CSC: " << status << std::endl;
        }

        // 作業バッファを解放
        cudaFree(dBuffer);

        cudaDeviceSynchronize();
    }




    cuMatSparse transpose(){
        //std::chrono::system_clock::time_point  start, end;
        //start = std::chrono::system_clock::now();

        cuMatSparse r(cols, rows, numVals);

        transpose(r);

        return r;
    }

    cuMat toDense() {
        cuMat r(rows, cols);

        // デフォルトのブロックサイズを 1x1 に設定
        int rowBlockDim = 1;
        int colBlockDim = 1;

        // BSR 行ポインタ配列と列インデックス配列のサイズ計算
        int* bsrRowPtrDevice;
        int* bsrColIndDevice;
        float* bsrValDevice;

        // GPU メモリの確保（適切なサイズを計算してください）
        cudaMalloc((void**)&bsrRowPtrDevice, (rows + 1) * sizeof(int));
        cudaMalloc((void**)&bsrColIndDevice, numVals * sizeof(int));
        cudaMalloc((void**)&bsrValDevice, numVals * sizeof(float));

        // 行列記述子の作成
        cusparseMatDescr_t descrA;
        cusparseCreateMatDescr(&descrA);
        cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);  // 行列のタイプを指定（必要に応じて変更）
        cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);  // インデックスの基準をゼロに設定

        cusparseStatus_t status = cusparseScsr2gebsr(
            cuHandle,
            CUSPARSE_DIRECTION_ROW,   // 行方向のブロックストレージ
            rows,                     // CSR 行列の行数
            cols,                     // CSR 行列の列数
            descrA,                   // CSR 行列の記述子
            csrValDevice,             // CSR 行列の値
            csrRowPtrDevice,          // CSR 行列の行ポインタ
            csrColIndDevice,          // CSR 行列の列インデックス
            bsrValDevice,             // 出力 BSR 行列の値
            bsrRowPtrDevice,          // 出力 BSR 行列の行ポインタ
            bsrColIndDevice,          // 出力 BSR 行列の列インデックス
            rowBlockDim,              // ブロック行サイズ
            colBlockDim);             // ブロック列サイズ

        if (status != CUSPARSE_STATUS_SUCCESS) {
            std::cerr << "toDense error" << std::endl;
        }

        // CUDA デバイス同期
        cudaDeviceSynchronize();

        // メモリの解放
        cudaFree(bsrRowPtrDevice);
        cudaFree(bsrColIndDevice);
        cudaFree(bsrValDevice);

        return r;
    }



    cuMatSparse toSparse(cuMat &a, int numVals){

            cuMatSparse r(a.rows, a.cols, a.rows);

            int *nnzPerRowColumn;
            cudaMalloc((void **)&nnzPerRowColumn, sizeof(int) * r.rows);
            int nnzTotalDevHostPtr = numVals;
            cusparseStatus_t status = cusparseSnnz(r.cuHandle, CUSPARSE_DIRECTION_ROW, r.rows,
                         r.cols, r.descr,
                         a.mDevice,
                         r.rows, nnzPerRowColumn, &nnzTotalDevHostPtr);
            if (status != CUSPARSE_STATUS_SUCCESS) {
                                    cout << "toSparse cusparseSnnz error" << endl;
                        }

            cudaDeviceSynchronize();


            status = cusparseSgebsr2csr(r.cuHandle, r.rows, r.cols,
                            r.descr,
                            a.mDevice,
                            r.rows, nnzPerRowColumn,
                            r.csrValDevice,
                            r.csrRowPtrDevice, r.csrColIndDevice);

            if (status != CUSPARSE_STATUS_SUCCESS) {
                        cout << "toSparse cusparseSgebsr2csr error" << endl;
            }
            cudaDeviceSynchronize();

            return r;
    }
};


#endif /* CUMATSPARSE_H_ */
