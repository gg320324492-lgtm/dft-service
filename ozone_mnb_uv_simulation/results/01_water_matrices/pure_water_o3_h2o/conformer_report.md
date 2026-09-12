# O₃·H₂O 局部水合构型搜索报告 (JOB-2026-0905-009)

**方法**: wB97X-D / def2-TZVP, full-derivative -D2 via make_mf_d2; SMD(H2O) for the solvation comparison

## 1. 构型池
- 初始构型数: 14（电荷 0 / 单重态）
- 原子序: O3 central, O3 term(+x), O3 term(-x), water O, water H1, water H2
- 覆盖: 末端氧供氢键 / 不同接近方向 / 共面与非共面 / 近远接触距离。

## 2. 气相构型搜索与去重
- 全局最低点: **c14_plus** (E = -301.87483581 Ha)
- 不同极小值数（3 kcal/mol 窗口）: 9
- 去重保留: ['c14_plus', 'c12_plus', 'c12_minus', 'c05', 'c11_minus', 'c11_plus', 'c09_plus', 'c09_minus', 'c06_plus']

| 构型 | 状态 | E_tot (Ha) | Δ(kcal/mol) | n_img | ν_min (cm⁻¹) | 氢键拓扑 | 保留? | 淘汰理由 |
|---|---|---|---|---|---|---|---|---|
| c01 | converged | -301.87409164 | 0.4674 | 1 | -110.00879331748357 | H-bonds: none | O_w contacts: term(+x) | False | duplicate of c06 (RMSD=0.092 A, dE=0.001 kcal/mol, same topology) |
| c02 | converged | -301.87309217 | 1.0946 | 2 | -66.72458373654675 | H-bonds: none | O_w contacts: term(+x) | False | saddle point: 2 imaginary mode(s) (nu_i = -66.72458373654675 cm-1), rel 1.095 kcal/mol -- not a minimum |
| c03 | converged | -301.87409244 | 0.4669 | 1 | -38.93926165819095 | H-bonds: none | O_w contacts: term(+x) | False | duplicate of c06 (RMSD=0.043 A, dE=0.001 kcal/mol, same topology) |
| c04 | converged | -301.87394234 | 0.5611 | 1 | -73.46099637072064 | H-bonds: none | O_w contacts: term(+x) | False | saddle point: 1 imaginary mode(s) (nu_i = -73.46099637072064 cm-1), rel 0.561 kcal/mol -- not a minimum |
| c05 | converged | -301.87483567 | 0.0005 | 0 | 63.20950129126652 | H-bonds: none | O_w contacts: central, term(+x), term(-x) | True |  |
| c06 | converged | -301.87409332 | 0.4664 | 1 | -100.55324896812675 | H-bonds: none | O_w contacts: term(+x) | False | saddle point: 1 imaginary mode(s) (nu_i = -100.55324896812675 cm-1), rel 0.466 kcal/mol -- not a minimum |
| c07 | converged | -301.87483493 | 0.001 | 1 | -58.81336939396613 | H-bonds: none | O_w contacts: central, term(+x), term(-x) | False | duplicate of c14 (RMSD=0.017 A, dE=0.001 kcal/mol, same topology) |
| c08 | converged | -301.8742978 | 0.3381 | 1 | -44.786570289004025 | H-bonds: none | O_w contacts: term(-x) | False | saddle point: 1 imaginary mode(s) (nu_i = -44.786570289004025 cm-1), rel 0.338 kcal/mol -- not a minimum |
| c09 | converged | -301.8748361 | 0.0003 | 1 | -109.06644971042479 | H-bonds: none | O_w contacts: central, term(+x), term(-x) | False | saddle point: 1 imaginary mode(s) (nu_i = -109.06644971042479 cm-1), rel 0.000 kcal/mol -- not a minimum |
| c10 | converged | -301.87279324 | 1.2822 | 5 | -104.55631008912333 | H-bonds: none | O_w contacts: none | False | saddle point: 5 imaginary mode(s) (nu_i = -104.55631008912333 cm-1), rel 1.282 kcal/mol -- not a minimum |
| c11 | converged | -301.87483503 | 0.001 | 1 | -118.11738214097015 | H-bonds: none | O_w contacts: central, term(+x), term(-x) | False | saddle point: 1 imaginary mode(s) (nu_i = -118.11738214097015 cm-1), rel 0.001 kcal/mol -- not a minimum |
| c12 | converged | -301.8748341 | 0.0015 | 1 | -92.33633094895566 | H-bonds: none | O_w contacts: central, term(+x), term(-x) | False | saddle point: 1 imaginary mode(s) (nu_i = -92.33633094895566 cm-1), rel 0.002 kcal/mol -- not a minimum |
| c13 | converged | -301.87148505 | 2.1031 | 5 | -135.58428765152934 | H-bonds: none | O_w contacts: none | False | saddle point: 5 imaginary mode(s) (nu_i = -135.58428765152934 cm-1), rel 2.103 kcal/mol -- not a minimum |
| c14 | converged | -301.87483654 | 0.0 | 1 | -52.13362954115357 | H-bonds: none | O_w contacts: central, term(+x), term(-x) | False | saddle point: 1 imaginary mode(s) (nu_i = -52.13362954115357 cm-1), rel 0.000 kcal/mol -- not a minimum |

单体: H₂O E=-76.43766761 Ha (n_img=0, ν_min=1619.6246814332176); O₃ E=-225.43390283 Ha (n_img=0, ν_min=784.5275618505681).

## 3. SMD(水) 比较（局部水合结构趋势，非水溶液结合常数）
- 溶剂模型: SMD(water)
- 保留的 SMD 极小值: ['c05', 'c14_plus', 'c09_plus', 'c11_minus', 'c06_plus']
- SMD 二次去重淘汰: {'c12_minus': 'collapsed onto SMD structure of c05 (RMSD small, same topology)', 'c12_plus': 'collapsed onto SMD structure of c09_plus (RMSD small, same topology)', 'c11_plus': 'collapsed onto SMD structure of c09_plus (RMSD small, same topology)', 'c09_minus': 'collapsed onto SMD structure of c05 (RMSD small, same topology)'}

| 构型 | 气相 E (Ha) | SMD E (Ha) | ΔE(SMD−gas) kcal/mol | 气相拓扑 | SMD 拓扑 |
|---|---|---|---|---|---|
| c05 | -301.87483567 | -301.89200416 | -10.773 | H-bonds: none | O_w contacts: central, term(+x), term(-x) | H-bonds: none | O_w contacts: term(+x) |
| c14_plus | -301.87483581 | -301.89200027 | -10.771 | H-bonds: none | O_w contacts: central, term(+x), term(-x) | H-bonds: none | O_w contacts: term(-x) |
| c09_plus | -301.87483543 | -301.89199846 | -10.770 | H-bonds: none | O_w contacts: central, term(+x), term(-x) | H-bonds: none | O_w contacts: term(-x) |
| c12_minus | -301.87483574 | -301.89199819 | -10.770 | H-bonds: none | O_w contacts: central, term(+x), term(-x) | H-bonds: none | O_w contacts: term(+x) |
| c12_plus | -301.87483574 | -301.89199540 | -10.768 | H-bonds: none | O_w contacts: central, term(+x), term(-x) | H-bonds: none | O_w contacts: term(-x) |
| c11_plus | -301.87483561 | -301.89199406 | -10.767 | H-bonds: none | O_w contacts: central, term(+x), term(-x) | H-bonds: none | O_w contacts: term(-x) |
| c11_minus | -301.87483567 | -301.89199054 | -10.765 | H-bonds: none | O_w contacts: central, term(+x), term(-x) | H-bonds: none | O_w contacts: term(+x) |
| c09_minus | -301.87483542 | -301.89198687 | -10.763 | H-bonds: none | O_w contacts: central, term(+x), term(-x) | H-bonds: none | O_w contacts: term(+x) |
| c06_plus | -301.87409391 | -301.89160826 | -10.990 | H-bonds: none | O_w contacts: term(+x) | H-bonds: none | O_w contacts: term(+x) |

## 4. 气相结合能与热化学
- Ghost-D2 排除审计: ghosts_excluded_ok=True (D2 差=0.0 Ha)

| 构型 | E_int(非CP) | E_int(CP) | BSSE | ZPE贡献 | ΔG_assoc(1atm) | ΔG_assoc(1M) |
|---|---|---|---|---|---|---|
| c14_plus | -2.049 | -1.531 | 0.518 | 1.070 | 4.695 | 2.801 |
| c12_plus | -2.049 | -1.529 | 0.520 | 0.956 | 2.989 | 1.095 |
| c12_minus | -2.049 | -1.529 | 0.520 | 0.956 | 2.990 | 1.096 |
| c05 | -2.049 | -1.548 | 0.501 | 1.097 | 4.788 | 2.894 |
| c11_minus | -2.049 | -1.537 | 0.512 | 1.125 | 4.893 | 2.999 |
| c11_plus | -2.049 | -1.536 | 0.513 | 1.129 | 4.910 | 3.016 |
| c09_plus | -2.049 | -1.543 | 0.506 | 1.119 | 4.783 | 2.889 |
| c09_minus | -2.049 | -1.543 | 0.506 | 1.119 | 4.782 | 2.888 |
| c06_plus | -1.583 | -1.376 | 0.207 | 1.186 | 5.051 | 3.158 |

标准态说明: O3 + H2O -> O3.H2O; 1 atm -> 1 M correction = -1.894 kcal/mol (applied to the gas RRHO association free energy).

## 5. CCSD(T)/aug-cc-pVTZ 单点相互作用能
- 基组: aug-cc-pVTZ
- 单体能量: O₃=-225.19466553512223 Ha, H₂O=-76.35750285266087 Ha

| 构型 | E_int CCSD(T) (kcal/mol) | T1 诊断 | 单参考? |
|---|---|---|---|
| c14_plus | -2.810 | 0.0643 | False |
| c12_plus | -2.810 | 0.0643 | False |

## 6. 方法局限
- Single-reference DFT (RKS); <S^2>=0 by construction, no spin contamination.
- Interaction energies are gas-phase; the SMD comparison is a LOCAL hydration-structure trend only and must NOT be read as an aqueous binding constant (no free-energy-of-solvation cycle was performed).
- geomeTRIC (delocalised internal coordinates) replaces berny because berny cannot build internal coordinates for the near-linear O_w-H...O hydrogen-bond chain; convergence assert_convergence=False, final geometry re-checked with a fresh make_mf_d2 (max|grad|<5e-5 Ha/Bohr).
- Boys-Bernardi CP correction uses PySCF ghost atoms (X-O / X-H); the project -D2 term excludes ghosts (verified) so no spurious dispersion pairs are introduced by the ghost basis.
- Standard states: gas RRHO 1 atm association free energy, and 1 M via the job-specified -1.894 kcal/mol correction for O3 + H2O -> O3.H2O.

## 7. 可否作为下一阶段初始结构
- 推荐: **c14_plus** — global gas-phase minimum (within the 3 kcal/mol distinct-minima set)
- 全部不同极小值: ['c14_plus', 'c12_plus', 'c12_minus', 'c05', 'c11_minus', 'c11_plus', 'c09_plus', 'c09_minus', 'c06_plus']
