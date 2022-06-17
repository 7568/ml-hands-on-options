# ml-hands-on-options
use ML methods to  hands optons

数据来源 https://www.gtarsc.com 深圳大学经济学院购买的版权 ，[股指期权 数据库说明书](https://github.com/7568/ml-hands-on-options/releases/download/v0.1/default.pdf)

数据的下载地址：https://github.com/7568/ml-hands-on-options/releases

两个表的基本信息如下：

| 下载表名 | 股指期权合约日交易基础表                                     |
| -------- | :------------------------------------------------------------ |
| 数据区间 | 2013-11-29 至 2022-06-01                                     |
| 选择代码 | 全部代码                                                     |
| 输出类型 | CSV格式（*.csv）                                             |
| 选择字段 | 证券ID[SecurityID] 交易日期[TradingDate] 交易代码[Symbol] 交易所代码[ExchangeCode]<br /> 合约简称[ShortName] 交易日状态编码[TradingDayStatusID] 标的证券ID[UnderlyingSecurityID] <br /> 标的证券交易代码[UnderlyingSecuritySymbol]   认购认沽[CallOrPut] 填充标识[Filling] <br />日开盘价[OpenPrice]  日最高价[HighPrice] 日最低价[LowPrice] 日收盘价[ClosePrice]  <br />日结算价[SettlePrice] 涨跌1[Change1] 涨跌2[Change2]  成交量[Volume] 持仓量[Position] <br /> 成交金额[Amount] 数据类型[DataType] |


| 下载表名 | 股指期权合约定价重要参数表                                   |
| -------- | :----------------------------------------------------------- |
| 数据区间 | 2013-11-29 至 2022-06-01                                     |
| 选择代码 | 全部代码                                                     |
| 输出类型 | CSV格式（*.csv）                                             |
| 选择字段 | 证券ID[SecurityID] 交易日期[TradingDate] 交易代码[Symbol] 交易所代码[ExchangeCode] <br /> 标的证券ID[UnderlyingSecurityID]  标的证券交易代码[UnderlyingSecuritySymbol]<br />合约简称[ShortName] 认购认沽[CallOrPut] 行权价[StrikePrice] 行权日[ExerciseDate] <br />收盘价[ClosePrice] 标的证券收盘价[UnderlyingScrtClose] 剩余年限[RemainingTerm]  <br />无风险利率(%)[RisklessRate] 历史波动率[HistoricalVolatility] 隐含波动率[ImpliedVolatility] <br />理论价格[TheoreticalPrice]  Delta[Delta] Gamma[Gamma] Vega[Vega] Theta[Theta] Rho[Rho] <br />连续股息率[DividendYeild] 数据类型[DataType] |

