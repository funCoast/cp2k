#include "frontend.h"
#include "IR.h"
#include "descriptor.h"
#include <vector>
void Frontend::Frontend::build()
{
    full_blocks.clear();
    right_partial_blocks.clear();
    noalign_blocks.clear();

    // 根据 dtype 计算 vector length
    int vl = IR::svl(dtype);

    int b = 0;
    // for (int b = 0; b < batch; ++b)
    {
        for (int i = 0; i < M; i += vl)
        {
            int row_step = std::min(vl, M - i);

            for (int j = 0; j < N; j += vl)
            {
                int col_step = std::min(vl, N - j);

                Block tile{.batch = b,
                           .row = i,
                           .column = j,
                           .row_step = row_step,
                           .column_step = col_step,
                           .trans_type = trans_type};

                if (row_step == vl && col_step == vl)
                {
                    full_blocks.push_back(tile);
                }
                else if (row_step == vl)
                {
                    right_partial_blocks.push_back(tile);
                }
                else if (col_step == vl)
                {
                    bottom_partial_blocks.push_back(tile);
                }
                else
                {
                    noalign_blocks.push_back(tile);
                }
            }
        }
    }
}
